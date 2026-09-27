# 07 — Multi-state design: `--dssp "parent/child_1/child_2"`

Spec for the third `--dssp` form, ported from
`salad/training/eval_split_change.py` (upstream, not in this repo - fetched at
`https://raw.githubusercontent.com/mjendrusch/salad/master/salad/training/eval_split_change.py`).
This is the multi-state protein design task described by
[Lisanza *et al.*, 2024](https://www.nature.com/articles/s41587-024-02395-w)
(Nature Biotechnology): design a *parent* protein with a specified secondary
structure which, split into an N-terminal and a C-terminal half, gives two
*child* proteins that adopt a *different* secondary structure from the
parent - while residues marked uppercase in the parent's DSSP string keep the
same *tertiary* structure (actual coordinates, not just a label) in the
matching child. Upstream's own README example (the Lisanza construct):

```bash
--dssp "_HHHHHHHHHHHHHlllHHHHHHHHHHHHHllleeeeeellleeeee_______eeeeellleeeeeelllHHHHHHHHHHHHHlllHHHHHHHHHHHHH_/_HHHHHHHHHHHHHLLLHHHHHHHHHHHHHLLLHHHHHHHHHHHHHHHHHHH_/_HHHHHHHHHHHHHHLLLHHHHHHHHHHHHHLLLHHHHHHHHHHHHH_"
```
(101/53/48 residues; `parent[:53]` pairs with `child_1`, `parent[53:]` with
`child_2`; the first two and last two helices of the parent are uppercase -
shared with the respective child.)

In addition to the usual denoising arguments (already ported: `--num_designs
--num_steps --out_steps --prev_threshold --config --params --timescale_pos`),
this reuses the single-state `--dssp` flag: a value containing `"/"` is
`"parent/child_1/child_2"` instead of one string.

---

## 1. Upstream's mechanism (read in full before porting)

Three independent, single-chain states (parent, child_1, child_2), each its
own length, run through the SAME denoising schedule in lockstep. Every step,
upstream's `step()` (`eval_split_change.py`) does, in this exact order, on
the *incoming* `pos` (the previous step's x0 prediction - all-zeros on step
0):

1. **Couple**: `SplitChange.update_pos(parent, child_1, child_2, ...)` -
   the crux of the feature, §2 below.
2. **Center** (`d["pos"] = center(d["pos"])`, CA-mean subtraction) - **not
   reproduced**, see §4.
3. **Noise + predict**, independently per state (three ordinary forward
   passes, one config, one set of params).

### 1.1 String parsing (`parse_parent`)

```python
def parse_parent(data: str):
    parent, child_1, child_2 = data.strip().split("/")
    parent_split = len(child_1)
    same_mask = np.array([c.isupper() for c in parent], dtype=np.bool_)
    parent = parse_dssp(parent.upper())
    child_1 = parse_dssp(child_1.upper())
    child_2 = parse_dssp(child_2.upper())
    move_op = SplitChange(same_mask, parent_split)
    return move_op, (parent, child_1, child_2)
```

Two independent things come out of one string:

* **Secondary-structure conditioning**, per state, from the `.upper()`-folded
  substring via the *ordinary* `DSSP_CODE = "LHE_"` table - identical
  mechanism to single-state `--dssp`, applied three times.
* **`same_mask`**: derived *only* from which characters are uppercase in the
  **parent** substring (`str.isupper()`; `"_"` is never upper, so an
  unspecified position can never be tertiary-coupled). This is the tertiary
  constraint's gate.

`parent_split = len(child_1)`, **not** `len(parent) // 2` - a literal
constant, not a computed midpoint. Upstream never checks
`len(parent) == len(child_1) + len(child_2)`; the port does (see §3) because
the coupling math needs a 1:1 residue correspondence between each parent
segment and its child to even be shape-valid, and a shape-mismatch several
calls downstream (inside the Kabsch fit) is a far more confusing failure than
rejecting it up front with the actual constraint named.

### 1.2 The coupling (`couple_part` / `SplitChange.update_pos`)

```python
def couple_part(x, y, mask):
    # y aligned to x
    y_at_x = index_align(y, x, zeros, ones*mask)
    # x aligned to y
    x_at_y = index_align(x, y, zeros, ones*mask)
    x = where(mask, y_at_x, x)
    y = where(mask, y * 0.5 + x_at_y * 0.5, y)
    return x, y

class SplitChange:
    def update_pos(self, parent, c_1, c_2, time):
        p_1, p_2 = parent[:self.parent_split], parent[self.parent_split:]
        m_1, m_2 = self.same_mask[:self.parent_split], self.same_mask[self.parent_split:]
        c_1, _ = couple_part(c_1, p_1, m_1)
        c_2, _ = couple_part(c_2, p_2, m_2)
        parent = concatenate((p_1, p_2))
        return parent, c_1, c_2
```

`index_align(x, y, ...)` (upstream and this port's own
`salad_torch/geometry/align.py`, already ported, same convention) rigidly
moves `x` onto fixed target `y`, fitting the Kabsch rotation from **only**
the masked residues (its `mask` argument gates the per-item mean/covariance
reduction inside `index_kabsch`) and then applying that one rigid transform
to *every* atom of `x`.

So `couple_part(x=child, y=parent_segment, mask)`:
* `y_at_x` = the parent segment, rigidly superimposed into the child's frame
  (fit computed from the masked residues only).
* `x_at_y` = the child, rigidly superimposed into the parent segment's frame.
* returns `(x_new, y_new)` where `x_new` = child with masked residues
  **replaced** by `y_at_x` (unmasked residues untouched - the `torch.where`
  is what actually preserves them, since `y_at_x`/`x_at_y` are full-length
  rigidly-moved copies, not just the masked slice); `y_new` = parent segment
  averaged 50/50 with `x_at_y` at masked residues.

**The coupling is a hard, one-directional coordinate override, not a loss
term**, and the direction is deliberately asymmetric:
`c_1, _ = couple_part(c_1, p_1, m_1)` **discards** the second return value
(the updated parent segment), and `parent = concatenate((p_1, p_2))`
reassembles the parent from its **original**, uncoupled segments. So: at
every step, each child's tertiary-coupled residues are forcibly overwritten
by the parent's current structure at those positions (superimposed into the
child's frame); the parent's own trajectory is **never** pulled toward
either child. This is reproduced exactly - see
`salad_torch/inference/multistate.py::update_split_change`'s docstring for
the same statement at the code site.

---

## 2. What the port implements

* **`salad_torch/inference/options.py`**: `MultiStateDSSP` (dataclass) +
  `parse_dssp_split(spec)` - the string parsing (§1.1), reusing the existing
  `parse_dssp` for each case-folded substring, plus the length-match
  validation upstream never had.
* **`salad_torch/inference/multistate.py`** (new module, mirroring how
  `--sym`'s geometry lives in `inference/symmetry.py` rather than in
  `options.py`): `couple_part` + `update_split_change` - direct, literal
  ports of §1.2, using this port's existing `geometry/align.py::index_align`
  (already ported, same "first arg moves, second is fixed" convention
  upstream's own docstring establishes - no changes needed there).
* **`scripts/generate.py`**: `run_multistate(args, config, model, device)` -
  a dedicated generation loop, dispatched from `main()` before
  `apply_precedence_chain` (which would otherwise choke trying to
  `DSSP_CODE.index("/")`). It builds three `(data, prev)` pairs via the
  existing `make_initial_data`, sets each state's own `dssp_condition`, and
  advances all three in lockstep via the **already-existing**
  `salad_torch.diffusion.sampling.sample_group` (built for N19's
  `--group_size`: "N independent designs, one shared clock, per-design
  state" - exactly the shape this task needs, with `n=3`). The only new
  per-step behavior is applying `update_split_change` to all three states'
  incoming `pos` at the top of `step_fn_group`, before each is noised/denoised
  - reusing the same `step_fn_group` seam `run_grouped`'s own multi-design
  batching already uses. **No changes to `sampling.py`, `model.py` or
  `fast.py`** - this is CLI-level composition of already-exported pieces,
  same cost tier as `--sym`/`--template` in `06_generation_flags.md`.

Three PDBs are written per design: `design_{n}_parent_{step}.pdb`,
`design_{n}_child_1_{step}.pdb`, `design_{n}_child_2_{step}.pdb` (one per
`--out_steps` entry), or three FASTA entries under `--sequence_only`.

---

## 3. Decisions this spec has to make that upstream's script doesn't force

### 3.1 `len(parent) == len(child_1) + len(child_2)`

Not checked upstream (see §1.1). The port's `parse_dssp_split` raises
`ValueError` naming the exact constraint if it fails, rather than reproducing
a downstream shape-mismatch. This is new validation, not a reproduced
upstream behavior - flagged as such in `parse_dssp_split`'s docstring.

### 3.2 Flag interactions - `run_multistate` rejects, upstream is silent

Upstream's own `eval_split_change.py` CLI (`parse_options` call) only ever
declares: `params, out_path, config, timescale_pos, timescale_seq, num_aa,
sym, merge_chains, variants, sym_mean, num_designs, num_steps, out_steps,
dssp, prev_threshold, cloud_std, sym_threshold, jax_seed`. Two different
situations follow from that:

* `--template`, `--order_agnostic`, `--group_size`, `--batch_size`, `--fast`,
  `--renoise`, `--sample_final` are **not in upstream's CLI for this script
  at all** - they are port-side additions to the *single-state* path
  (`06_generation_flags.md`). Genuinely combining any of them with three
  coupled, independently-lengthed trajectories is new design work outside
  this task's scope (a 3-state renoise loop, a 3-state compiled/CUDA-graph
  path, deciding which of three states a template applies to, etc.) -
  `run_multistate` rejects them outright (`SystemExit`) rather than
  half-implementing.
* `--sym`, `--merge_chains`, `--cloud_std` **are** declared in upstream's
  `parse_options` call for this script, but the script's body never reads
  `opt.sym`/`opt.merge_chains`/`opt.cloud_std` anywhere - dead, copy-pasted
  CLI surface from the shared de-novo script. Upstream's true behavior is
  silent inertness; **this port instead rejects them explicitly** if passed
  non-default. Decision, not a bug: a user who typed `--merge_chains`
  expecting something to change is better served by a clear error than by
  silent, upstream-faithful inertness indistinguishable from a port bug.

`--depth_adjust`/`--compile`/`--seed`/`--jax_seed`/`--sequence_only` compose
normally and are not restricted (the first two are already applied to
`config`/`model` before `run_multistate` is even called).

### 3.3 Per-step centering (`center()`) - not reproduced, matching existing precedent

Upstream's `step()` recentres each state's `pos` on its own CA mean
(`d["pos"] = center(d["pos"])`) between the coupling and the noising. This
port's existing **single-state** loop (`scripts/generate.py`'s ordinary path,
`salad_torch/modules/model.py::apply_diffusion`) does not reproduce
upstream's per-step centering either - it was never flagged as a gap in
`06_generation_flags.md`, and the model's own noising/prediction is
translation-invariant (only relative geometric features are read), so
omitting it does not change the model's behavior, only the absolute
coordinate frame of the written-out trajectory. `run_multistate` follows the
same existing precedent for consistency rather than introducing centering
that no other part of this port's sampling loop has. Documented here, not
silently dropped.

---

## 4. Evidence this works (not just "a PDB exists")

`.agent-research/results/N035-multistate/` runs the Lisanza construct
end-to-end and checks two independent things per design:

1. **Secondary structure**: a CA-only P-SEA assignment (reusing
   `.agent-research/results/N011-fast-preset/ss.py::psea`, per instruction -
   not reimplemented) per state, scored against the requested `--dssp`
   string (ignoring `"_"` positions) for parent, child_1, child_2
   independently.
2. **Tertiary coupling**: Kabsch-superposed CA RMSD between each child and
   its parent segment, restricted to `same_mask` positions, **compared
   against an UNCOUPLED control** (the same three DSSP strings, generated as
   three *independent* single-state runs with `update_split_change` never
   applied) - both arms share the same secondary-structure target at those
   positions, so a low RMSD from secondary-structure agreement alone is not
   sufficient evidence; the coupled/uncoupled gap is what isolates the
   tertiary mechanism specifically.

See that directory's `RESULTS.md` for the actual numbers.
