# 06 — Generation-loop flags (symmetry, template, renoise, order-agnostic, misc)

Spec for the flags of `salad/training/eval_noise_schedule_benchmark.py` that
`scripts/generate.py` does not yet cover. Everything here is additive on top
of the existing de-novo CLI (`config`, `params`, `num_aa`, `num_designs`,
`num_steps`, `out_steps`, `prev_threshold`, `timescale_pos`, `cloud_std`,
`out_path`) and the existing model-level conditioning (`dssp_condition`,
`dssp_mean`, `aa_condition`, motif `dmap`/`omap`/`dmap_mask`, multi-chain,
cyclic — all already wired through `salad_torch/modules/condition.py`).

All file:line references below are to
`salad/training/eval_noise_schedule_benchmark.py` in the upstream JAX tree
unless stated otherwise. The writer does not have that file; every algorithm
needed to implement a flag is spelled out below in full, not just cited — the
citations are for audit/traceability only.

Existing port hooks this spec builds on (do not re-derive these, use them):

* `salad_torch/diffusion/sampling.py::sample(step_fn, data, prev, cfg, *,
  generator, post_step_fn, final_step_fn)` — `post_step_fn(pos, raw_t) -> pos`
  runs on `data["pos"]` (the x0 prediction) right after every step, before it
  is written back and before the out-step snapshot is taken. `final_step_fn`
  replaces `step_fn` (not composes with it) on out-steps only.
* `salad_torch/modules/model.py::apply_diffusion(config, data, *, generator,
  noise, seq_noise)` — the noising step, already factored out as a free
  function, and `StructureDiffusionInference.predict` /
  `StructureDiffusionPredict` — the denoise-only half, already exposed as
  `model.predict` in `scripts/generate.py:70`. `StructureDiffusionInference`
  composes them; nothing stops a caller from composing them differently.
* `salad_torch/io/pdb.py::write_pdb(path, snapshot, *, chain_index=None,
  apply_residue_mask=False)` / `snapshot_to_protein` — already takes an
  **independent** `chain_index` override for exactly the pre-merge / template
  chain-labelling use case (its docstring says so).
* `salad_torch/config.py::DotDict` — `__getattr__ = dict.get`, so **any
  config key that is never assigned reads as `None`**, silently, everywhere.
  This is used below to establish that a whole branch (`config.symm`) is dead.

---

## 0. Precedence chain the CLI must reproduce

Upstream builds `num_aa` / `resi` / `chain` / `base_chain` / `cyclic_mask` in
several places that run in a fixed order and each unconditionally overwrite
the previous stage's values for the fields they touch (`eval_...:490-550`).
Get the order wrong and flags that look independent silently stop applying.
In order:

1. `parse_num_aa(opt.num_aa)` → `resi`, `chain`, `is_cyclic`, `cyclic_mask`.
   `base_chain = chain` (a copy, taken **before** step 2).
2. `--merge_chains`: `chain = zeros_like(chain)`. `base_chain` is untouched.
3. `--dssp <string>` (a literal string, not `"none"`/`"random"`): if given,
   `num_aa = len(dssp_string)`, `resi = arange(num_aa)`,
   `base_chain = chain = zeros(num_aa)`. **This silently discards
   `--num_aa`, `--merge_chains` and any cyclic spec** — the dssp string's
   length is now the only thing that determines the design's size, and the
   result is a single unnumbered chain.
4. `--template <path>,<constring>`: if given, `num_aa = dmap.shape[0]`,
   `resi = arange(num_aa)`, `base_chain = chain = chain_index` (from the
   constring's `:`-separated chain groups — see §2). This runs **after**
   step 3, so if both `--dssp` and `--template` are given, the template's
   size/chain silently wins and step 3's `num_aa` is discarded — if the two
   sizes disagree, downstream tensors (`dssp_condition` built for the old
   `num_aa`) will not broadcast, i.e. a hard shape-mismatch crash, not a
   graceful error. Do not special-case this — reproduce the crash, don't add
   validation upstream never had.
   `is_cyclic = False`, `cyclic_mask = zeros(num_aa, bool)`: **`--template`
   always disables cyclic conditioning**, unconditionally.
5. `--order_agnostic`: `resi = -ones(num_aa)`, `chain = zeros(num_aa)`. Runs
   **last**, after both 3 and 4, so it always wins for those two fields.
   `base_chain` (the PDB-output chain) is **not** touched here, so combining
   `--order_agnostic` with `--template` still writes the template's chain
   letters to the output PDB even though the network sees one merged,
   unordered chain.

A consequence worth stating once, generally: `--merge_chains` is a no-op
whenever `--dssp <string>` or `--template` is also given, because both of
those unconditionally reassign `chain` (and `base_chain`) after
`--merge_chains` has already run.

Port this as a straight-line sequence of dict/tensor reassignments in the CLI
script, in this exact order, before `make_initial_data`.

---

## 1. Symmetry — `--sym`, `--sym_mean`, `--sym_threshold`, `--replicate_count`,
`--screw_angle`, `--screw_radius`, `--screw_translation`

### 1.1 What actually runs, and what doesn't

Three call sites exist in the model/eval-script for "symmetrize a tensor of
positions". Only two of them are reachable from this CLI; the third is dead
in every released config. Do not spend time on the dead one.

| Site | What it symmetrizes | Reachable? |
|---|---|---|
| `config.sym_noise(pos)` inside the noising step, on `pos_noised` (the network's noised *input*) | the noised x_t the trunk actually sees | **yes** — set by `--sym` |
| `symm=c.symm` argument of `update_positions(...)` / `update_positions_precomputed(...)`, inside every structure-update layer of the trunk (`noise_schedule_benchmark.py:2740-2796`, called at `:2162,2238,2297,2388,2467`) | the trunk's per-layer local-frame position update, before it's projected back to world coordinates | **no — dead code, see 1.2** |
| The eval loop's own post-hoc call, on `update["pos"]` (the model's x0 prediction) after every step (`eval_...:611-619`), and once on `init_pos` before the loop starts (`eval_...:537-544`) | the sampler's world-frame trajectory | **yes** — gated by `--sym` (+ `--sym_threshold` for the per-step case) |

So there are exactly **two** things to implement: a `sym_noise`-equivalent
hook into the noised input, and a post-hoc hook on world-frame positions
(used twice: once for `init_pos`, once per step). Both use the same
`screw_replicate` function underneath, with different fixed arguments.

### 1.2 `config.symm` / `update_positions(symm=...)` — DEAD, do not port

`config.symm` is read via `DotDict.__getattr__ = dict.get`, i.e. it is `None`
unless something explicitly sets `config.symm = ...`. Nothing in
`eval_noise_schedule_benchmark.py`'s `__main__` block (or in any of the
`default_vp*` / `multimotif_vp` configs the port supports) ever assigns
`config.symm`. So for every `--sym` invocation of this script,
`update_positions(..., symm=None)` runs with the symmetrizer branch
(`if symm is not None: pos_update = symm(pos_update)`) permanently skipped.
**Do not implement this hook.** It would require new model-side code (a
`symm` argument threaded through every structure-update call in the trunk)
for a code path this eval script never exercises.

Also note: `salad/inference/symmetry.py`'s `Screw` class (a more general,
class-based reimplementation of the same replicate/couple/select math,
imported by `salad/inference/__init__.py`) is **not used by
`eval_noise_schedule_benchmark.py` at all** — that script has its own,
textually-duplicated `screw_replicate`/`cyclic_centering`/`rot_x/y/z`
functions (the ones specified below). `Screw` is dead code for this CLI.
Ignore `salad/inference/symmetry.py` entirely; nothing in this spec depends
on it.

### 1.3 Core geometry: `rot_y`, `cyclic_centering`, `screw_replicate`

`rot_y(angle)` (angle in **radians**):

```
rot_y(a) = [[ cos(a), 0, -sin(a)],
            [   0,    1,    0   ],
            [ sin(a), 0,  cos(a)]]
```

`cyclic_centering(x, docenter=True)`: if `docenter` is false, return `x`
unchanged; else subtract the mean of column-index-1 (the CA atom — atom
order is N, CA, C, ...) from every atom: `x - mean(x[:, 1], axis=0)`,
broadcast over all atoms and all residues in `x`.

`screw_replicate(x, count, angle, radius, translation, chain_center, time,
mean)` — `x` is `(L, A, 3)` where `L` must be evenly divisible by `count`
(**input contract**: `--num_aa` must specify exactly `count` chains, ideally
of equal length; `size = L // count` uses integer floor division, so if the
total is not evenly divisible the last `L - size*count` residues are simply
never read or written by this function — a silent no-op on the remainder,
not an error):

```
size = L // count
rot = rot_y(angle)
rot_pow = [I, rot, rot@rot, ..., rot^(count-1)]      # length count, built by
                                                       # repeated left-mult
# ---- 1. canonicalize one subunit from the current L residues ----
if mean:
    acc = 0
    for i in range(count):
        seg = x[i*size:(i+1)*size]
        centered = cyclic_centering(seg, docenter=True)      # subtract seg's own CA mean
        acc += centered @ rot_pow[i].T                        # undo subunit i's rotation
    first = acc / count                                        # Procrustes-style average
else:
    i = count // 2
    first = cyclic_centering(x[i*size:(i+1)*size], docenter=True) @ rot_pow[i].T

# ---- 2. optional self-clash repulsion nudge on the CA column only ----
def cost(ca):     # ca: (size, 3), the representative subunit's CA coordinates
    d = sqrt(max((ca[:, None] - ca[None, :]) ** 2, 1e-3).sum(-1))
    switch = max(1 - ((d - 2) / 8) ** 6, 1e-6) / max(1 - ((d - 2) / 8) ** 12, 1e-6)
    return switch.mean()

grad = d(cost)/d(first[:, 1])                # autograd, NOT a closed form — see 1.3.1
grad = where(isnan(grad), 0, grad)
if chain_center:
    first = first + time * 5 * grad[:, None]  # broadcasts the (size,3) CA nudge to all atoms

# ---- 3. place the subunit off-axis and replicate around the screw axis ----
first = first + [radius, 0, 0]                # shift along +X, away from the Y rotation axis
replicates = [ (first @ rot_pow[i]) + i * [0, translation, 0]  for i in range(count) ]
result = concat(replicates, axis=0)           # back to (L, A, 3)
return result - mean(result[:, 1], axis=0)     # re-center the WHOLE assembly on its CA mean
```

Notes for the writer:

* `translation != 0` gives true screw (helical) symmetry — a rotation about Y
  composed with a rise along Y per subunit. `translation == 0` degenerates to
  plain cyclic (Cₙ point-group) symmetry. `--replicate_count` is `count`.
  `--screw_angle` is in **degrees** on the CLI and must be converted to
  radians (`angle_rad = screw_angle / 180 * pi`) before calling this.
  `--screw_radius` is `radius`, `--screw_translation` is `translation`.
* The `cost`/`grad` step is a smooth pairwise CA-CA clash penalty (a custom
  ratio-of-two-switches, not a physical potential — reproduce the exact
  formula, both the `1e-6` floors and the `(d-2)/8` scaling are load-bearing
  constants, not tunables) and its gradient nudges the representative subunit
  away from *its own* self-clashes before radial placement. It only fires
  when `chain_center=True`.

#### 1.3.1 Autograd inside a `torch.no_grad()` sampling loop

`scripts/generate.py`'s `step_fn` wraps the whole model call in
`torch.no_grad()`, and the post-hoc symmetrizer runs on `data["pos"]`
produced from inside that block. `torch.autograd.grad` requires an
autograd-tracked leaf tensor and a grad-enabled context; under the ambient
`no_grad()` neither holds. The symmetrizer implementation must locally
re-enable autograd around this one computation, e.g.:

```python
with torch.enable_grad():
    ca = first[:, 1].detach().requires_grad_(True)
    loss = cost(ca)
    grad, = torch.autograd.grad(loss, ca)
grad = torch.nan_to_num(grad, nan=0.0)
```

This is a correctness requirement, not a style choice: a naive port that
forgets the `enable_grad()` scope will either raise ("does not require
grad") or (if `requires_grad_` is set but the outer `no_grad()` still wins)
silently produce `None`/zero gradients, which for `chain_center=False`
(the common case) is unobservable but for `chain_center=True` would silently
disable the repulsion nudge.

### 1.4 Hook A — `init_pos` symmetrization (setup, once)

`eval_...:537-544`. After building `init_pos` (all zeros,
`(num_aa, atom_count, 3)`) and before entering the design loop, if `--sym`:

```python
init_pos = screw_replicate(init_pos, count=replicate_count,
                            angle=screw_angle_rad, radius=screw_radius,
                            translation=screw_translation,
                            chain_center=False, mean=(sym_mean == "True"))
```

`chain_center` is hardcoded `False` here (no repulsion nudge on the initial,
all-zero cloud — there is nothing to repel from yet). `mean` **does** honor
`--sym_mean`. Since `init_pos` is all zeros, `screw_replicate` on it is
almost a no-op geometrically (the "canonicalize" step averages/selects zero
subunits, the repulsion grad on coincident points is whatever the
`1e-3`-floored distance formula gives at zero separation, then the result is
placed at `radius` and rotated) — it mainly establishes the initial
`count`-fold radial arrangement of the (still zero-valued) noise cloud before
the first noising step spreads it out. Port this as a one-line addition in
the CLI right after `make_initial_data`, overwriting `data["pos"]`.

### 1.5 Hook B — `sym_noise` on the noised input (every network call)

`eval_...:501-508` sets `config.sym_noise` to a closure over `screw_replicate`
with **`chain_center=False, mean=False` hardcoded** — note `mean=False`
regardless of `--sym_mean`. This is an upstream inconsistency worth flagging
explicitly: `--sym_mean` controls hooks A and C below but is silently ignored
for this one, which always uses the "pick the middle subunit" branch
(`mean=False`), never the Procrustes-average branch. Reproduce this exactly
— do not wire `--sym_mean` into this call.

The model applies `sym_noise` to `data["pos_noised"]` immediately after
`apply_diffusion` produces it, before the trunk sees it
(`noise_schedule_benchmark.py:458-459`, inside
`StructureDiffusionInference.__call__`). The port's
`StructureDiffusionInference.forward` fuses `apply_diffusion` and
`predict` with no seam to hook into — but it doesn't need one, because both
halves are already separately exported. **Do not modify `model.py`.**
Instead, when `--sym` is set, the CLI's `step_fn` should stop calling the
fused `model(...)` and instead compose the two exported pieces directly:

```python
def step_fn(data, prev, generator=None):
    with torch.no_grad():
        noised = apply_diffusion(config, data, generator=generator)
        data = dict(data, **noised)
        if sym_active:                       # config.sym_noise is not None, i.e. --sym
            data["pos_noised"] = screw_replicate(
                data["pos_noised"], count=replicate_count,
                angle=screw_angle_rad, radius=screw_radius,
                translation=screw_translation,
                chain_center=False, mean=False)
        return model.predict(data, prev)
```

`model.predict` is `StructureDiffusionPredict`, already exposed as
`model.predict` in `scripts/generate.py:70` (used there for
`torch.compile`). This makes hook B **cheap**: zero new model code, just a
different `step_fn` body in the CLI, reusing `apply_diffusion` (already
`__all__`-exported from `salad_torch/modules/model.py`) and
`model.predict`.

### 1.6 Hook C — post-hoc per-step symmetrization on the x0 prediction

`eval_...:610-620`. This is exactly the case `sample()`'s `post_step_fn`
docstring already names ("the hook for the `--sym` screw symmetrisation").
Wire it as:

```python
def post_step_fn(pos, raw_t):
    if raw_t > sym_threshold:      # --sym_threshold, default 0.0
        return screw_replicate(pos, count=replicate_count,
                                angle=screw_angle_rad, radius=screw_radius,
                                translation=screw_translation,
                                chain_center=(raw_t > 0.5), time=raw_t,
                                mean=(sym_mean == "True"))
    return pos
```

`chain_center=(raw_t > 0.5)` and `time=raw_t`: the repulsion nudge (§1.3) is
only active in the high-noise first half of the trajectory (`raw_t > 0.5`)
and its magnitude scales with `raw_t` itself — i.e. it is strongest early and
fades to zero as `raw_t → 0.5`, and is off entirely for `raw_t <= 0.5`.
`--sym_threshold` (default `0.0`) additionally gates the *whole* symmetrizer
call by `raw_t`: at the default, since the reproduced `raw_t` schedule never
reaches exactly 0 (see `sampling.py`'s documented off-by-one), this
symmetrizes after every single step. Raising `--sym_threshold` (e.g. to
`0.5`) disables post-hoc symmetrization once the trajectory drops below that
`raw_t`, letting the low-noise/late refinement steps evolve without a forced
symmetry constraint — this is the intended use of the flag: enforce symmetry
while the topology is being laid down, relax it for final local geometry
clean-up.

This needs no new hook in `sampling.py` — pass this closure as `post_step_fn`
to the existing `sample(...)` call.

### 1.7 Cost tier & interactions

All of §1.4–1.6 are **cheap**: pure CLI/setup-level composition of existing
exported pieces (`apply_diffusion`, `model.predict`, the `post_step_fn`
hook) plus one new self-contained geometry function (`screw_replicate` +
its two small helpers). No changes to `salad_torch/modules/*` are required.
The only subtlety is the `enable_grad()` scope inside a `no_grad()` sampler
(§1.3.1) and reproducing the `mean=False` inconsistency in hook B (§1.5).

Interactions: `--sym` requires `--num_aa` to be given as `--replicate_count`
colon-separated segments (§1.3's divisibility contract) — it is incompatible
in spirit with `--template` and `--dssp <string>` (both override `num_aa`
to a single unsegmented size per §0) and with `--order_agnostic` (which
merges all chains to 0, defeating the per-subunit segmentation
`screw_replicate` relies on via `size = L // count`). Upstream does not
guard against combining them; neither should the port — just know that the
combination produces geometrically meaningless output rather than an error.

---

## 2. `--template` / `--template_aa`

### 2.1 Constring parsing

The flag value is `"<pdb_path>,<constring>"`. The constring has one character
per residue plus `:` separators between chains, e.g. `"FFF__FFF:FF"` = chain
0 has 8 residues (positions 1-3 and 6-8 "F"ixed/conditioned, 4-5 free), chain
1 has 2 residues, both fixed:

```python
conmask, chain_index = [], []
chain = 0
for c in constring:
    if c == "F":
        conmask.append(True); chain_index.append(chain)
    elif c == ":":
        chain += 1
    else:                      # any other character (conventionally "_")
        conmask.append(False); chain_index.append(chain)
size = len(conmask)
dmap_mask = conmask[:, None] & conmask[None, :]     # (size, size) bool
```

`dmap_mask[i, j]` is true only where **both** `i` and `j` are `"F"`-flagged —
the distance-map conditioning is only trustworthy between two motif residues
that both actually came from the template structure.

### 2.2 Reading the template PDB (new code — the port has no PDB reader yet)

`salad_torch/io/pdb.py` only writes PDBs (`to_pdb`/`write_pdb`); there is no
`from_pdb_string`/`read_pdb`. This is genuinely new code, not a rewiring of
something that already exists. Upstream's `from_pdb_string` (`aflib`,
`protein.py:188-207`) is a thin wrapper around Biopython's `PDBParser`
feeding into `_from_bio_structure` (`protein.py:101-166`); reproduce that
recipe with Biopython (`Bio.PDB.PDBParser`) or an equivalent minimal parser:

```
parse with Bio.PDB.PDBParser(QUIET=True)
models = list(structure.get_models())
if len(models) != 1: raise ValueError   # multi-model PDBs are refused
for chain in models[0]:
    for res in chain:
        if res.id[2] != " ": raise ValueError   # insertion codes are refused
        one_letter = rc.restype_3to1.get(res.resname, "X")     # non-standard -> X/UNK
        restype_idx = rc.restype_order.get(one_letter, rc.restype_num)  # UNK -> 20
        pos = zeros(37, 3); mask = zeros(37); bfac = zeros(37)
        for atom in res:
            if atom.name not in rc.atom_types: continue         # unknown atoms ignored
            pos[rc.atom_order[atom.name]] = atom.coord
            mask[rc.atom_order[atom.name]] = 1.0
            bfac[rc.atom_order[atom.name]] = atom.bfactor
        if mask.sum() < 0.5: continue        # residue with zero recognized atoms: dropped
        append aatype=restype_idx, atom_positions=pos, atom_mask=mask, ...
chain_index: map each Bio chain-id letter to a 0-based integer via
    {cid: n for n, cid in enumerate(sorted(unique(chain_letters)))}
    (i.e. alphabetical by chain letter, NOT file order, matching np.unique)
```

All the residue-constant tables (`rc.restype_3to1`, `rc.restype_order`,
`rc.restype_num`, `rc.atom_types`, `rc.atom_order`) already exist verbatim in
`salad_torch/io/residue_constants.py` — this is purely assembling them, no
new constants needed. The output should be the port's own `Protein`
dataclass fields (`atom_positions (N,37,3)`, `aatype (N,)`, `atom_mask
(N,37)`, `chain_index`, `residue_index` — the parsed PDB's own author-assigned
residue numbers, but see below, they end up unused) — i.e. build a `Protein`,
not a new type.

The parsed PDB's own `residue_index` and `chain_index` are **not** what
determines the model's relative-position features later — see §2.4 (`resi`
is reset to `arange(num_aa)` unconditionally once a template is present).
`chain_index` from the *constring*, not from the PDB file, is what's used —
the constring's chain groups need not correspond 1:1 to chains in the PDB
file at all; the template PDB's own chain boundaries are irrelevant once its
coordinates have been extracted into `atom14`/`ncacocb` sequentially in
file-residue order and re-indexed against `conmask`/`chain_index` positions.

### 2.3 From parsed structure to `dmap`/`omap`/`aa_condition`

Reusing existing port functions (`atom37_to_atom14`, `positions_to_ncacocb`,
`extract_aa_frames`, all already present):

```python
atom14, _ = atom37_to_atom14(aatype_parsed, atom_positions_parsed, atom_mask_parsed)
ncacocb = positions_to_ncacocb(atom14)         # (n_provided, 5, 3): N,CA,C,O,pseudo-CB
cb_provided = ncacocb[:, 4]                     # (n_provided, 3)

size = len(conmask)                             # = number of residues in the CONSTRING, not the PDB
cb = zeros(size, 3);            cb[conmask] = cb_provided
ncacocb_full = zeros(size, 5, 3); ncacocb_full[conmask] = ncacocb
aatype = full(size, 20);         aatype[conmask] = aatype_parsed        # 20 = MASK/unconditioned

frames, _ = extract_aa_frames(ncacocb_full)     # Rigid; frames.rotation: (size, size?) -- see below
rot = frames.rotation                            # (size, 3, 3), plain tensor (port's Rigid, not JAX Vec3Array)
inv_rot = rot.transpose(-1, -2)                  # rotation matrices are orthogonal: inverse == transpose
omap = inv_rot[:, None] @ rot[None, :]           # (size, size, 3, 3), broadcast batched matmul
omap = omap.reshape(size, size, 9)

dmap = norm(cb[:, None] - cb[None, :], dim=-1)   # (size, size)
```

Important: `extract_aa_frames` is called on the **full, size-length**
`ncacocb_full` array (masked-out positions are all-zero, so their frames are
degenerate/garbage) — the garbage rows/columns of `omap`/`dmap` for
non-`"F"` residues are exactly what `dmap_mask` (§2.1) exists to hide from
the model; they are never read. `aa_condition` is `aatype`
(size-length, sentinel 20 where not conditioned) and is only fed to the
model at all if `--template_aa` is also set (see below) — but `dmap`/`omap`/
`dmap_mask` are always built and always used once `--template` is given,
independent of `--template_aa`.

Return `(aatype, dmap, omap, dmap_mask, chain_index)` from this parsing
function, matching the eval script's `parse_template` return signature.

### 2.4 Wiring into the design loop

`eval_...:529-535, 568-573`. Once a template is parsed:

```python
num_aa = dmap.shape[0]                    # = len(constring), overrides --num_aa entirely
resi = arange(num_aa)                     # synthetic, sequential — the template PDB's own
                                           # residue numbers (with any gaps) are discarded
base_chain = chain = chain_index          # from the constring, see §0 step 4
is_cyclic = False
cyclic_mask = zeros(num_aa, bool)

data["dmap"] = dmap
data["omap"] = omap
data["dmap_mask"] = dmap_mask
if template_aa:
    data["aa_condition"] = aa_condition   # aatype from §2.3; sequence identity conditioning
```

`data["dmap"]`/`"omap"`/`"dmap_mask"` are already consumed by
`salad_torch/modules/condition.py::prepare_condition` exactly as built here
(`"the *mask* is the gate, not the map: a dmap without a dmap_mask is
ignored"` — its docstring already documents this contract) — **no changes
needed there**. `aa_condition`, likewise, is already read by
`prepare_condition` (`if "aa_condition" in data: aa = data["aa_condition"]`).
So wiring this flag in is CLI-level work (build the tensors per §2.2-2.3,
put them in `data`) on top of conditioning machinery that already exists and
is already tested.

### 2.5 `c.use_omap` — the orientation-map embedding is not implemented, and no released config needs it

`prepare_condition` always builds and forwards `pair_condition["omap"]`
(zeros if no template, real values if templated), matching upstream, which
is why §2.3-2.4 above are unconditional. But upstream's own trunk only
*consumes* `omap` when `c.use_omap` is true
(`noise_schedule_benchmark.py:2061-2065`, upstream's own comment: `"optionally
embed orientation maps (not used in the manuscript)"`), and:

```
grep 'use_omap' salad/modules/config/noise_schedule_benchmark.py
```
shows `use_omap=True` only on `*_omap`-suffixed experimental configs, **none
of which** the port supports (`default_vp`, `default_vp_scaled`,
`default_ve_scaled`, `multimotif_vp` in `salad_torch/config.py` all leave
`use_omap` unset ⇒ `None` ⇒ falsy). So for every checkpoint this port can
load, the omap embedding branch is provably unreachable — building `omap`
(§2.3) is still required (`aa_condition`/`dmap`/`dmap_mask` reuse the same
parsing pass and are load-bearing), but the model-side embedding of it is a
**dead end, not worth implementing now**: it would need a new
`self.linear_omap = HkLinear(9, pair_size, bias=False)` in
`DiffusionPairFeatures.__init__` (`salad_torch/modules/blocks.py:82`,
inserted immediately after `self.linear_dmap`, gated on `c.use_omap` exactly
like the existing `self.multi_motif` → `self.linear_dmap_mask` gating right
above it), and a matching forward-pass line inserted between the existing
`pair = pair + self.linear_dmap(dmap_rbf)` and `flags = torch.where(...)`
(`salad_torch/modules/blocks.py:129-130`):

```python
if c.use_omap:
    omap_masked = torch.where(pc["dmap_mask"][..., None], pc["omap"],
                              torch.zeros((), device=pos.device, dtype=pair.dtype))
    pair = pair + self.linear_omap(omap_masked)
```

and a corresponding entry in `convert/haiku_to_torch.py::_pair_feature_names`
— which, note, **already has this exact `if use_omap: names.append
("linear_omap")` branch** (`convert/haiku_to_torch.py:107-108`), i.e. the
checkpoint-conversion side was already written in anticipation of this, only
the model-forward side is missing. Leave it missing. Implement this only if
a future config with `use_omap=True` is added; do not add speculative
untested code for it now.

### 2.6 Cost tier

Reading the template PDB into a `Protein` (§2.2) is the only genuinely new,
non-trivial piece — a Biopython-based parser, medium effort, no precedent in
the port. Everything downstream of that (§2.3's tensor algebra, §2.4's
wiring) is cheap, reusing existing port functions and an already-implemented
conditioning path. §2.5 (`use_omap`) is correctly a dead end for now.

### 2.7 Interactions

See §0 for the precedence over `--dssp`/`--merge_chains`/`--order_agnostic`.
Additionally: `--template` disables cyclic conditioning unconditionally
(`is_cyclic = False`), so `--template` combined with a cyclic `--num_aa` spec
(the `cN:...` syntax) silently drops the cyclic flag — this is not a bug to
fix, it's upstream's documented precedence (§0 step 4), just note it.

---

## 3. `--renoise`

### 3.1 What it does

`eval_...:625-644`. At every requested `--out_steps` step, after the normal
snapshot is taken, if `--renoise` is set: run **9 independent** extra
"partial re-denoise" passes starting from the just-produced x0 estimate,
each one re-entering the diffusion process at `raw_t = 1 - 0.6` (i.e.
`s = int(num_steps * 0.6)`) and stepping forward to the current out-step's
`s = step`, **with self-conditioning forced off at every one of those
sub-steps** (not just gated by `--prev_threshold` — `prev = init_prev`
unconditionally, every sub-step):

```python
snapshots = [snapshot(data)]             # the ordinary, already-computed out-step result
if renoise:
    for _ in range(9):
        d = copy(data)                    # see §3.2 — the port must NOT alias the live loop state
        for s in range(int(num_steps * 0.6), step):
            raw_t = 1 - s / num_steps
            d["t_pos"] = timescale_pos(raw_t) * ones_like(d["t_pos"])
            d["t_seq"] = timescale_seq(raw_t) * ones_like(d["t_seq"])
            update, _ = step_fn(d, init_prev, generator)   # ALWAYS the base step_fn, never final_step_fn
            d["pos"] = update["pos"]
            d["seq"] = update["aa"].argmax(-1)
            d["atom_pos"] = update["atom_pos"]
            d["aatype"] = update["aatype"]
        snapshots.append(snapshot(d))
```

If `step < int(num_steps * 0.6)` (an early out-step, before the 60% mark),
the inner `range` is empty and that renoise pass is an identical copy of the
un-renoised snapshot — not an error, just a no-op pass. Each of the 9 passes
draws fresh randomness (noise inside `step_fn`'s `apply_diffusion`, plus any
sampling randomness) and diverges from the others as soon as the inner loop
runs at least once.

Write all 10 snapshots (1 original + 9 renoise) to separate files. Upstream's
exact naming (`eval_...:645-671`, reproduce it): the loop that writes files
does `for idr, data in enumerate(results):` where `results[0]` is the
un-renoised original — so **file index `0` is the original, not a renoise
pass** — and `results[1..9]` are the 9 renoise passes, giving filenames
`result_{design}_{idr}_{step}.pdb` for `idr` in `0..9` (10 files per
out-step) or, under `--sequence_only`, FASTA headers
`>design_{design}_{idr}_{step}`.

### 3.2 An upstream bug: do not reproduce the state aliasing

Upstream's inner loop mutates the **same** `data` dict object the outer
per-step loop owns (`data["pos"] = pos` at `eval_...:638` is an in-place key
assignment on the live dict, not a copy), for every inner iteration of every
one of the 9 passes. The line `for idr, data in enumerate(results):`
(`eval_...:645`) then **rebinds the name `data` in the enclosing scope** to
`results[-1]` — Python `for` loops don't scope their loop variable, so after
this file-writing loop finishes, the outer scope's `data` now refers to the
**last renoise pass's final, self-conditioning-off, partially-refined
state**, not the properly-continued original trajectory.

If `step == out_steps[-1]` this is harmless (the loop `break`s immediately
after). But if `--out_steps` names **more than one** step (e.g.
`--out_steps 200,400`) and `--renoise` is on, every out-step except the last
one silently forks the rest of the main trajectory through its 9th renoise
pass's output instead of continuing normally. This is a scoping/aliasing
accident (mutable shared dict + a `for` variable outlives its loop), not a
principled design choice that shapes published numbers the way the `raw_t`
off-by-one does — reproducing it would mean deliberately threading global
mutable state through the port's `sample()` in a way that fights its whole
design (which already copies `data` explicitly per step for exactly this
reason). **Do not reproduce it.** Implement renoise so each pass operates on
its own copy and the main trajectory's `data`/`prev` are unaffected by
having run renoise at an intermediate out-step. Flag this in a code comment
at the call site so a future bit-exact-parity effort knows it's a deliberate
deviation, not an oversight.

### 3.3 Interactions and cost

* Always uses the **base** `step_fn`, never `final_step_fn` — `--sample_final`
  has no effect inside a renoise pass even though it gates the outer
  per-step call.
* `--prev_threshold` has no effect inside a renoise pass (self-conditioning
  is unconditionally off throughout).
* `--timescale_seq`/`--timescale_pos` are reused as configured for the main
  loop.
* Cost: with the CLI defaults (`num_steps=500`, one out-step at `400`), each
  renoise pass runs `400 - 300 = 100` extra forward steps, times 9 passes =
  **900 extra forward calls per design**, i.e. roughly a 3x wall-clock
  multiplier on top of the base 500-step run for that one design. This is
  expensive; surface it in `--help`/docs.
* Implementation: this is loop-shaped, not model-shaped — implement as a
  small helper next to `sample()` in `salad_torch/diffusion/sampling.py`
  (it needs `parse_timescale`, already imported there) or directly in the
  CLI script; either is fine, no new model code.

---

## 4. `--order_agnostic`

Two independent pieces: an input-side change (network never sees residue
order) and an output-side change (the written PDB is re-threaded by
geometry instead of by the network's internal indexing).

### 4.1 Input side (setup, once — §0 step 5)

`resi = -ones(num_aa)`, `chain = zeros(num_aa)`, applied once before the
design loop, after any `--dssp`/`--template` overrides.

Check this is safe against `GF.sequence_relative_position`
(`salad_torch/geometry/features.py:101-`): with every `resi` equal to `-1`,
`rel = resi[i] - resi[j] = 0` for every pair; `clip(rel, -count, count) +
count = count` — a single, fixed, in-range bin (never negative, never
out-of-bounds for the one-hot). With `chain` all zero, `same_chain` is
always true, so the cross-chain sentinel bin is also never used. The
network genuinely receives **zero** sequence-order information (every pair
looks like "adjacent, same chain") — this is the intended effect, and it is
safe (no `IndexError`/out-of-range one-hot risk) to implement as a direct
value substitution, no special-casing needed in `sequence_relative_position`
itself.

Side effect worth stating plainly: because `chain` is also zeroed, this has
the same chain-merging effect as `--merge_chains` in addition to erasing
`resi` — combining `--order_agnostic` with an intentionally-multi-chain
design merges the chains too (see §0's note on `--merge_chains` interacting
with later overrides; this is the same mechanism, just from a different
flag). If `--num_aa` used the cyclic `cN:` syntax, note that
`sequence_relative_position`'s cyclic-wrap branch computes per-chain lengths
from `chain` (`index_count(chain, ...)`) — with `chain` now all zero, that
"length" becomes the whole merged sequence length rather than each original
chain's length, which does not correspond to the original per-chain cyclic
wrap intent. Not a case worth specifically guarding against; just don't be
surprised by it.

### 4.2 Output side: `reorder_amino_acids` (post-hoc, PDB-writing only)

`eval_...:409-449`. Runs **once**, on the final `Protein` object right before
`to_pdb`, and only affects the written file's residue order/numbering — it
never feeds back into the sampling loop's own `data`. Uses atom37-format
positions (the `N` and `C` backbone atoms).

```python
N = atom_positions[:, atom_order["N"]]      # (L, 3)
C = atom_positions[:, atom_order["C"]]      # (L, 3)
C_to_N_dist = norm(C[:, None] - N[None, :], dim=-1)   # (L, L)
C_to_N_dist[eye(L, bool)] = inf                        # exclude self
adjacency = C_to_N_dist < 2.5                          # bool (L, L): peptide-bond-like contact

visited = zeros(L, bool)
weight  = ones(L, int)          # placeholder; every visited residue's weight is overwritten
next_of = argmax(adjacency, axis=1)      # first True column per row (or 0, meaningless, if none)
stop    = ~adjacency.any(axis=1)         # true = "no outgoing C->N bond": a chain terminus

count, index, vinloop = 0, 0, []
while True:
    if index in vinloop:                          # closed a cycle within this walk
        if visited.all(): break
        index = argmax((~visited) & (~stop))       # jump to a fresh, non-terminal start
        vinloop = []
    visited[index] = True
    vinloop.append(index)
    weight[index] = count
    count += 1
    if stop[index]:                                 # hit a terminus: end this walk
        if visited.all(): break
        index = argmax((~visited) & (~stop))
        count += 1
    else:
        index = next_of[index]

order = argsort(weight)
new_residue_index = weight[order]         # 0, 1, 2, ... in visitation order (with gaps at walk boundaries)
new_chain_index = zeros(L)                # every reordered residue is written as chain 0
reordered = Protein(atom_positions[order], aatype[order], atom_mask[order],
                    new_residue_index, new_chain_index, b_factors[order])
```

This is a genuine graph walk (follow the C→N "next" pointer, restart at a
fresh non-terminal residue whenever the current walk closes a cycle or hits
a dead end), not a sort — port it as-is, including the two separate
"restart" call sites (top-of-loop cycle-closure, and the stop-residue
branch) and the `count += 1` that only happens in the second restart branch.
`new_chain_index` is unconditionally all-zero: this function assumes/produces
a single output chain regardless of how many chains went in.

**Robustness note — flag, and add a safety net the upstream code lacks.**
`argmax((~visited) & (~stop))` can only select a residue that is *both*
unvisited *and* has at least one outgoing C→N bond. A residue that is fully
isolated — no outgoing bond (`stop=True`) **and** never any other residue's
incoming `next_of` target — can never be selected as a fresh start (it's
excluded by `~stop`) and can never be reached by a walk (nobody points to
it). If such a residue exists and is never visited, `visited.all()` never
becomes true, and `argmax` over an all-`False` row returns index `0` every
time; if residue 0 is not itself such an isolated residue, this causes the
algorithm to re-walk an already-visited chain segment indefinitely — an
**unbounded infinite loop**, not a graceful failure. This is plausible for
a raw diffusion sample (a local geometry defect breaking backbone
connectivity is exactly the kind of thing de-novo generation occasionally
produces), so it is not a purely theoretical concern. Reproducing an
unbounded hang is not useful behavior to port faithfully. Add a hard
iteration cap (e.g. `2 * L` total `while` iterations) and raise a clear
error past it, rather than reproducing the hang.

### 4.3 Cost and interactions

Both pieces are cheap: §4.1 is two tensor assignments in the CLI setup, §4.2
is a small, pure, CPU-only post-processing function operating on the final
`Protein`/snapshot (same layer as `snapshot_to_protein`/`to_pdb`, no model
code involved). See §0 for the precedence vs. `--dssp`/`--template`, and
§4.1 for the `--merge_chains`-equivalent side effect on `chain`.

---

## 5. Remaining flags

### 5.1 `--merge_chains` — cheap, mostly already built

Already covered in §0 (step 2: `chain = zeros_like(chain)`, `base_chain`
unaffected). The only other piece: `write_pdb`/`snapshot_to_protein` already
accept an explicit `chain_index=` override
(`salad_torch/io/pdb.py:158-172`), whose docstring literally says *"should
be the pre-`--merge_chains` assignment... Defaults to the snapshot's own
`chain_index`, which is the post-merge one"* — i.e. this flag is already
fully supported at the I/O layer; the only CLI-level work is (a) keep
`base_chain` and the merged `chain` as two separate tensors through the
sampling loop, and (b) pass `chain_index=base_chain` to `write_pdb`.

One additional effect worth documenting (not obvious from the tensor alone):
`prepare_condition`'s default `chain_contacts = (chain[:,None] !=
chain[None,:]) & (batch[:,None]==batch[None,:])` (`condition.py:134`) is
derived from `chain`. Under `--merge_chains`, `chain` is uniformly zero, so
`chain_contacts` is uniformly `False` — the "these two residues are on
different chains" pairwise flag channel (`pair_condition["flags"][...,0]`,
embedded via `linear_flags`) goes completely dark in addition to the relpos
effect. `--merge_chains` therefore removes cross-chain information from
*two* separate feature channels, not just relative position.

### 5.2 `--sequence_only` — needs a corrected `decode_sequence`, upstream's is broken

`eval_...:645-654` (FASTA branch) vs. `:655-671` (PDB branch), selected by
this flag; both write into the `results`/`idr` loop from §3.1, with the same
naming convention (`sequences.fa`, append mode, one shared file across the
whole run — every design and every out-step's sequence accumulates into a
single `{out_path}/sequences.fa`, opened in `"at"` append mode and
explicitly `flush()`ed after every write). Reproduce the shared-file/append
behavior; do not create one file per design.

**Upstream bug, don't reproduce it.** The eval script imports
`decode_sequence` from `salad.data.allpdb`
(`from salad.data.allpdb import decode_sequence`, `eval_...:27`), whose
signature is `decode_sequence(x: np.ndarray) -> str` — **one argument**. The
call site passes two: `decode_sequence(data["aatype"],
jnp.ones_like(chain, dtype=jnp.bool_))` (`eval_...:652-653`). This is a hard
arity mismatch — calling it with `--sequence_only True` raises `TypeError:
decode_sequence() takes 1 positional argument but 2 were given` in the
upstream script as shipped. There is no alternate 2-argument
`decode_sequence` in scope (there is one, `eval_s2s.py:82`, but it is not
the one imported here). Do not try to reproduce this crash, and do not go
looking for a "correct" 2-argument version to match bit-for-bit — implement
a working decoder directly against the port's own tables, which are
confirmed consistent with what the model's `aatype` actually uses (checked:
`allpdb.py`'s own `AA_CODE = "ARNDCQEGHILKMFPSTWYVX"` **does** match
`salad_torch/io/residue_constants.py::restypes` (`"ARNDCQEGHILKMFPSTWYV"`,
i.e. the canonical AlphaFold order, not alphabetical) with `X` appended for
the sentinel/UNK index 20 — so there is no ordering bug, only the arity bug):

```python
def decode_sequence(aatype: Tensor) -> str:
    table = rc.restypes_with_x            # already exists, restypes + ["X"]
    return "".join(table[i] for i in aatype.tolist())
```

The second argument upstream passes is always `ones_like(chain, dtype=bool)`
— an always-true, effectively-unused mask (every residue is written
regardless) — so dropping it entirely is behavior-preserving, not merely
"close enough".

### 5.3 `--sample_final` (+ temperature) — needs new model-side code

`eval_...:512-515` builds a **second** model instance from a deep-copied
config with `config.sample_aa = True` and `config.temperature = 0.1`
(**hardcoded** — there is no separate `--temperature` CLI flag upstream,
despite `--sample_final` being the flag that activates temperature-based
sampling). This second model (`model_final`) replaces `model` **only** on
out-steps (`eval_...:606-609`) — exactly what the port's `sample()`
`final_step_fn` parameter already exists for (its docstring: *"the
`--sample_final` second model, invoked instead of `step_fn` on output
steps"*).

The blocker: `config.sample_aa = True` routes the amino-acid decoder through
`AADecoder.sample` (upstream `noise_schedule_benchmark.py:1100-1106`,
`:1524-1541`), and the port's equivalent,
`salad_torch/modules/decoders.py::AADecoder.sample`, is **currently a stub
that raises `NotImplementedError`** (its docstring: *"unreachable for
default_vp, where config.sample_aa is unset"*) — true for `default_vp`
itself, but `--sample_final` is exactly the case that makes it reachable.
This must be implemented; it is genuine new model code, not CLI wiring.

Exact algorithm (`AADecoder.sample`, upstream, `:1524-1541`) — a categorical,
random-order, single-residue-per-iteration autoregressive decode that
**reruns the full decoder stack from scratch every iteration**:

```python
def sample(self, aa, local, pos, resi, chain, batch, mask, temperature, generator):
    L = aa.shape[0]
    aa = full_like(aa, 20)                         # start fully masked
    for _ in range(L):                              # exactly L iterations, one residue revealed each time
        logits, _ = self(aa, local, pos, resi, chain, batch, mask)   # full forward, current partial aa
        logits = logits / (temperature or 0.1)       # temperature default matches c.temperature or 0.1
        u = rand(L, generator=generator)              # one uniform float per residue
        u = where(aa == 20, u, inf)                    # only currently-masked residues are eligible
        index = argmin(u)                               # uniformly pick ONE masked residue to reveal
        sampled = categorical(logits, dim=-1, generator=generator)  # draws L independent categorical
                                                                     # samples; only `sampled[index]` is used
        aa = aa.clone(); aa[index] = sampled[index]
    return aa
```

Notes:
* This costs `L` full forward passes through the AA-decoder stack (not the
  structure trunk) per design on every out-step where `--sample_final` is
  active — expensive for long sequences, but it's exactly what upstream
  does; there is no cheaper faithful alternative.
* The per-iteration categorical draw samples logits for **all** `L`
  positions even though only the chosen `index`'s sample is kept — port this
  literally (draw the full `(L, 20)` categorical every iteration) rather
  than "optimizing" it to draw a single categorical sample at `index`, since
  that would consume a different amount of randomness per iteration and
  change the sampled sequence under a fixed seed. Bit-for-bit reproduction
  against the JAX reference is impossible regardless (different PRNG
  algorithms), but keep the *shape* of what's drawn faithful for internal
  consistency and for anyone later diffing against a from-scratch torch
  re-implementation.
* Wire the CLI: build a second `StructureDiffusionPredict`/`Inference`
  sharing the same loaded parameters but with `config.sample_aa = True` (a
  `deepcopy` of the config, matching upstream's `final_config`), and pass a
  `final_step_fn` closure that calls into this sampling path, to `sample()`.
  `--sample_final`'s temperature is not exposed as its own CLI flag upstream
  — hardcode `0.1` to match, or add a `--sample_final_temperature` flag
  defaulting to `0.1` as a port-side enhancement (does not change default
  behavior either way).

### 5.4 `--dssp` / `--dssp_mean` string parsing

`parse_dssp_mean` (`eval_...:159-166`) / `parse_dssp` (`:168-174`); both are
called once at CLI-parse time and return either `None`, a fixed array, or a
**callable** that gets invoked fresh per design (`eval_...:576-585`: `if
isinstance(dssp_mean, Callable): data["dssp_mean"] = dssp_mean()` /
`data["dssp_condition"] = dssp(num_aa)`).

```
DSSP_CODE = "LHE_"        # 0=loop(L), 1=helix(H), 2=strand(E), 3=sentinel/none(_)
                          # matches salad_torch/data/dssp.py's own convention exactly

parse_dssp_mean(s):
    "none"   -> None
    "random" -> the callable random_dssp_mean (§5.4.1), invoked with no args each design
    else     -> array of 3 floats, "loop,helix,sheet", parsed directly with NO renormalization —
                the port's Condition.forward overrides its own computed per-chain composition
                whenever set_condition["dssp_mean"] is given (condition.py:91-94), using this
                triple AS-IS, so it is the caller's responsibility to pass values that already
                sum to 1; nothing downstream enforces or corrects that.

parse_dssp(s):
    "none"   -> None
    "random" -> the callable random_dssp (§5.4.1), invoked as random_dssp(num_aa) each design
    else     -> array of len(s) ints via [DSSP_CODE.index(c) for c in s]; each character must be
                one of "L","H","E","_" (case-sensitive; anything else raises ValueError from
                str.index, which is the correct/desired failure mode, not something to guard)
```

Note the `--dssp <string>` (non-`"none"`/`"random"`) branch is the one that
overrides `num_aa` per §0 step 3 — `len(s)` becomes the design's residue
count.

#### 5.4.1 `random_dssp_mean` / `random_dssp` — lower priority, full algorithm for completeness

These generate a *fresh random* secondary-structure specification per design
(useful for topology-diversity sweeps, not for reproducing a specific
target). They are independent draws — `random_dssp` calls
`random_dssp_mean()` internally but only to get fresh helix/sheet
*fractions*; it does not share state with a separately-configured
`--dssp_mean random`, so the string-level DSSP and the composition target
are never actually coupled even when both are set to `"random"` for the same
design.

```
random_dssp_mean(max_loop=0.5):
    helix = uniform(0, 1); sheet = 1 - helix
    loop = uniform(0, 1) * max_loop
    return [loop, (1 - loop) * helix, (1 - loop) * sheet]     # NOT renormalized to sum 1 either

random_dssp(count, p=0.5):
    loop_frac, helix_frac, sheet_frac = random_dssp_mean()     # loop_frac discarded below
    helix = int(helix_frac * count); if helix < 6: helix = 0
    min_h, max_h = 1, helix // 6; min_h = min(min_h, max_h)
    num_helices = 0 if helix == 0 else (min_h if min_h == max_h else randint(min_h, max_h))  # numpy: high-exclusive
    sheet = int(sheet_frac * count); if sheet < 8: sheet = 0
    loop = count - (helix + sheet)
    min_s, max_s = 2, sheet // 4
    num_sheets = 0 if sheet == 0 else (min_s if min_s == max_s else randint(min_s, max_s))
    helices = [6] * num_helices          # each helix segment starts at fixed length 6
    sheets  = [4] * num_sheets           # each sheet segment starts at fixed length 4
    loops   = [0] * (num_helices + num_sheets + 1)     # N+1 filler gaps around N structured segments
    while sum(sheets)  < sheet:  sheets[randint(num_sheets)]  += 1   # distribute remainder randomly
    while sum(helices) < helix:  helices[randint(num_helices)] += 1
    while sum(loops)   < loop:   loops[randint(0, len(loops))] += 1
    # render each segment to a character string:
    #   helix segment of length n: "_"+"H"*(n-2)+"_" with prob (1-p), else "_"*n  (fully unconditioned)
    #   sheet segment of length n: "_"+"E"*(n-2)+"_" with prob (1-p), else "_"*n
    #   loop  segment of length n: "L"*n with prob p, else "_"*n
    structured = shuffle(helix_strings + sheet_strings)     # random order of the structured elements
    dssp_string = loop_strings[0] + "".join(s + l for s, l in zip(structured, loop_strings[1:]))
    return parse_dssp(dssp_string)     # -> int array via DSSP_CODE, length == count (if the
                                        # random draws above summed exactly to `count`, which they do
                                        # by construction: helix + sheet + loop == count always)
```

`randint(a, b)` above follows numpy's convention (`b` exclusive); the
`min == max` special-casing exists specifically to avoid `randint(a, a)`,
which numpy raises on. Upstream mixes `numpy`'s global RNG (`np.random.*`)
and Python's stdlib `random` module (`random.random()`, `random.shuffle`)
for different sub-decisions within this one function — since bit-exact
parity with the JAX reference is already impossible for a torch port
(different PRNG algorithms throughout), there is no need to preserve which
specific RNG backs which draw; just keep each decision an independent random
draw as written. This whole section is **lower priority** than §1-4 — it's
a topology-diversity convenience generator, not core numerical machinery; a
reasonable interim port can stub `"random"` to `NotImplementedError` and
ship `"none"` and explicit strings/arrays first.

### 5.5 `--depth_adjust` — cheap, works via existing checkpoint-loading order

`eval_...:499-500`: `if opt.depth_adjust != "none": config.diffusion_depth =
int(opt.depth_adjust)`, applied **before** the model is constructed and
**before** parameters are loaded. This matters because of how depth is
represented: upstream's `hk.LayerStack`-based trunk stores per-layer weights
stacked along a leading axis of length `diffusion_depth`; the port's
`DiffusionStack` (`salad_torch/modules/blocks.py:284-313`) instead builds an
`nn.ModuleList` of `config.diffusion_depth` separate `DiffusionBlock`
modules, and `load_model`/`load_diffusion_stack`
(`salad_torch/convert/haiku_to_torch.py:208-220, 282-306`) loads layer `i`'s
weights with `for layer in range(config.diffusion_depth): ... layer=layer`
— i.e. it slices the checkpoint's stored per-layer array at index `layer`
for `layer` in `0..config.diffusion_depth-1`.

Consequently: **setting `config.diffusion_depth = N` before constructing and
loading the model, for `N <=` the checkpoint's native depth (6, for the
released `default_vp` checkpoints), reproduces upstream's early-exit
ablation exactly** — the model is built with only the first `N` trunk
blocks, and only the first `N` of the checkpoint's trained per-layer weight
sets are loaded; layers `N..5` are simply never built/run. This needs **no
model-side code change at all** — it's purely "set the config attribute
before `get_config(...)`'s result is used to build+load the model" in the
CLI, exactly mirroring the ordering upstream uses.

`N >` the checkpoint's native depth is **not supported**: there are no
extra trained per-layer weights to load, and the port's
`build_state_dict(..., layer=layer)` will index past the end of the stored
array and raise (an `IndexError`-shaped failure) rather than silently
fabricating untrained layers. This is the correct behavior to leave as-is —
do not add a fallback that invents random weights for `layer >= 6`.

### 5.6 `--jax_seed` — already fully ported, just rename/note the mapping

`scripts/generate.py` already has `--seed` → `torch.Generator(device=...)
.manual_seed(args.seed)`, and it's already used the way upstream uses
`jax_seed`: **one** generator/key created once, then threaded through and
advanced continuously across the *entire* run (all designs sequentially),
not reset per design (`scripts/generate.py:86-103`, matching upstream's
single `key = jax.random.PRNGKey(opt.jax_seed)` at the top of `__main__`,
split every step and every design, never re-seeded). No further work is
needed beyond, if desired, aliasing `--jax_seed` as a documented synonym for
`--seed` for anyone porting a command line verbatim. Note explicitly in
`--help`/docs that **no seed value gives bit-identical output between the
JAX and PyTorch backends** — the two frameworks' PRNG algorithms are
unrelated, so `--seed`/`--jax_seed` only gives reproducibility *within* one
backend, not parity *across* them.

---

## 6. Summary table

| Flag(s) | Cost tier | New model code? | Dead end? |
|---|---|---|---|
| `--sym`, `--sym_mean`, `--sym_threshold`, `--replicate_count`, `--screw_*` | cheap | no (composes existing `apply_diffusion`/`model.predict`/`post_step_fn`) | `config.symm`/`update_positions(symm=)` path and `salad/inference/symmetry.py::Screw` — yes, both dead, don't port |
| `--template` (dmap/omap/dmap_mask/aa_condition) | medium | no (conditioning already wired) — but needs a **new PDB reader** (no precedent in the port) | the `use_omap` trunk embedding — yes, dead for all 4 supported configs, recipe given for later |
| `--template_aa` | cheap (once `--template`'s parser exists) | no | no |
| `--renoise` | cheap (loop-shaped, reuses `step_fn`) | no | no — but reproduce the *fixed* (non-aliasing) semantics, not upstream's dict-aliasing bug (§3.2) |
| `--order_agnostic` | cheap | no | no — but add an iteration cap to `reorder_amino_acids` that upstream lacks (§4.2) |
| `--merge_chains` | cheap | no (`chain_index=` param already exists on `write_pdb`) | no |
| `--sequence_only` | cheap | no | upstream's imported `decode_sequence` — arity bug, implement a corrected 1-line version instead (§5.2) |
| `--sample_final` (+temperature) | medium | **yes** — `AADecoder.sample` is currently `NotImplementedError` in the port | no, this is required for the flag to do anything |
| `--dssp` / `--dssp_mean` (fixed string/array or `"none"`) | cheap | no | no |
| `--dssp` / `--dssp_mean` `"random"` mode | medium, low priority | no | no, but defer-able |
| `--depth_adjust` | cheap | no (pure config-then-load ordering) | no |
| `--jax_seed` | already done | — | — |
