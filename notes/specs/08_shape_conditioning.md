# 08 — Shape conditioning: `--blobs` / `--blob_scale` / `--replace_threshold`

Spec for the one flag family `salad/training/eval_ve_shape.py` adds on top of
the plain de-novo CLI (`notes/specs/06_generation_flags.md`'s base surface:
`config`, `params`, `num_designs`, `num_steps`, `out_steps`, `prev_threshold`,
`timescale_pos`). Upstream source (fetched 2026-09-27,
`raw.githubusercontent.com/mjendrusch/salad/master/salad/training/eval_ve_shape.py`
— there is no local copy in this repo; the default branch is `master`, not
`main`). `salad/inference/shape.py` is *not* a dependency of this script —
it is two bare `import jax`/`import jax.numpy as jnp` lines and nothing else,
dead code, safely ignored.

Upstream's own example:

```
python salad/training/eval_ve_shape.py --params params/default_ve_scaled-200k.jax \
  --out_path out/ --blobs letters/s_shape.csv --blob_scale 1.8 \
  --num_designs 50 --num_steps 500 --out_steps 400
```

`letters/{a,d,l,s}_shape.csv` (fetched from the same tree) are upstream's
shipped example CSVs — every row in all four files has `num_aa=100`. That is
load-bearing, not a coincidence (§2).

---

## 0. What upstream's `parse_blob` does

```python
def parse_blob(c, path):
    init_pos = []
    with open(path) as f:
        next(f)
        for line in f:
            *xyz, num_aa = line.strip().split(",")
            num_aa = int(num_aa)
            xyz = np.array([float(i) for i in xyz], dtype=np.float32)
            xyz = np.repeat(xyz[None, None], num_aa, axis=0)
            xyz = np.repeat(xyz, 5 + c.augment_size, axis=1)
            init_pos.append(xyz)
    init_pos = np.concatenate(init_pos, axis=0)
    num_aa = init_pos.shape[0]
    resi = np.arange(num_aa)
    chain = np.zeros_like(resi)
    is_cyclic = False
    cyclic_mask = np.zeros_like(chain, dtype=np.bool_)
    return num_aa, init_pos, resi, chain, is_cyclic, cyclic_mask
```

Each CSV row places `num_aa` consecutive residues at `(x, y, z)`: **every
atom of every one of those residues starts at the exact same literal 3-vector**
— not spread out yet. `init_pos` (shape `(total_num_aa, 5+augment_size, 3)`)
becomes the network's `data["pos"]` before the first noising step, in place
of the de-novo script's all-zero cloud. `resi`/`chain` are **flat**: one
contiguous `arange`, one all-zero chain — `parse_blob` never calls
`parse_num_aa` and has no per-segment 50-residue gap, no cyclic support, and
no per-blob chain numbering (all blobs collapse into ONE unnumbered output
chain, regardless of how many rows the CSV has).

`opt.blob_scale *= init_pos` runs in `__main__`, after `parse_blob` returns —
multiplies every coordinate. Order relative to computing `num_aa`/`resi`/
`chain` from `init_pos.shape` doesn't matter (a scalar multiply never
changes shape); the port folds the multiply inside `parse_blobs` itself.

Two quirks reproduced deliberately, not fixed:

* `next(f)` unconditionally discards line 1 — there is no check it actually
  reads `"x,y,z,num_aa"`. A headerless CSV silently loses its first blob.
* `*xyz, num_aa = line.strip().split(",")` — a row with other than 4
  comma-separated fields is not rejected at parse time; it silently changes
  how many elements land in `xyz` (a non-3-vector coordinate raises later,
  inside `np.array`/the broadcast against `init_pos`, not here).

## 1. Why the point cloud alone is "VE noise arranges residues in a normal
distribution around each location"

`apply_diffusion`'s edm branch (`salad_torch/diffusion/forward.py`,
`diffuse_coordinates_edm`) computes
`pos_noised = data["pos"] + t * centred_normal_noise` for
`config.diffusion_kind == "edm"` (`default_ve_scaled`'s `diffusion_kind`).
With `data["pos"]` seeded from `parse_blobs`, at the first step (`raw_t=1`,
`t_pos = timescale_pos(1)`, upstream's default `"ve(t)"` giving a large sigma
near `sigma_max=80`) this is *exactly* "each blob's coordinate + a wide,
~isotropic Gaussian offset" — the normal-distribution-around-each-location
behaviour the task describes, and it requires **zero new model code**: it is
already what `apply_diffusion` does to whatever is in `data["pos"]`. Nothing
in `salad_torch/modules/` needed to change for this half of the feature.

This is also why the feature is VE-model only: `apply_diffusion` only takes
this branch for `config.diffusion_kind == "edm"`, which only `default_ve_scaled`
sets (`config.preconditioning == "edm_scaled"`, `salad_torch/config.py`).
Every other released config takes the `vp`/`vpfixed`/`flow` branch instead
(`diffuse_atom_cloud`, a per-structure-CA-spread or fixed-`sigma_data` cloud
std, unrelated to `data["pos"]`'s actual values) — the blob coordinates would
still seed `data["pos"]`, but the noising step would not read them as an
anchor the same way, and the mechanism this spec relies on (§1) simply
would not apply. **The port raises `SystemExit` if `--blobs` is combined
with a non-edm config** rather than silently running a channel that doesn't
mean what it means for VE, since nothing in this port's model code adapts
the blob mechanism to any other preconditioning.

## 2. The per-step "hard replace" — `--replace_threshold`

The second, independent mechanism. Upstream's `__main__`, inline (no name):

```python
pos = update["pos"]
pos_centers = pos[:, 1].reshape(-1, 100, 3).mean(axis=1, keepdims=True)
pos_centers = jnp.repeat(pos_centers, 100, axis=1).reshape(-1, 3)
if raw_t >= opt.replace_threshold:
    pos = pos - pos_centers[:, None, :] + init_pos
data["pos"] = pos
```

`init_pos` here is the same fixed, blob-derived tensor from §0 — never
updated over the run. `pos[:, 1]` is the CA column. Reshaping into
`(-1, 100, 3)` **chunks the flat residue axis into blocks of exactly 100**,
computes each block's mean CA (its centroid), then subtracts that centroid
from every atom of every residue in the block and adds back `init_pos` for
that residue (not the block's mean of `init_pos` — each individual residue's
own assigned blob coordinate).

Net effect per residue `i` in block `b`:
`new_pos[i] = (pos[i] - mean_CA(pos[b])) + init_pos[i]` — i.e. the model's
own *local* geometry (its offset from its block's current centroid) is
preserved, but the block's placement in world space is forcibly re-anchored
so its centroid equals the mean of that block's assigned blob coordinates.
Without this, nothing else in the network structurally "knows" where the
blobs are beyond the noising statistics (§1), and a 500-step trajectory can
drift the shape away from the requested layout; with it, drift is corrected
back to the blob anchor after every qualifying step.

**`chunk=100` is hardcoded upstream, not derived from the CSV's own
`num_aa` column.** It is a bare assumption that every blob spans exactly 100
residues — true of all four shipped example CSVs (§ header). The port
(`replace_to_blob_anchor`, `salad_torch/inference/blobs.py`) reproduces this
as a literal constant, **not** generalised to "one chunk per blob": if the
total residue count is not a multiple of 100, `.reshape(-1, 100, 3)` raises
`RuntimeError`, exactly as upstream's own `reshape` would. This is not
caught or validated — per this port's standing convention (see spec 06 §0,
§2.1) of reproducing an upstream crash rather than adding graceful handling
upstream never had. A CSV whose rows are not all 100 (or another correctly-
dividing-the-total value) will still *parse*, but `--replace_threshold`'s
default of 0.0 means the crash happens on the very first step of every run.

`--replace_threshold` (default `0.0`, matching upstream): gates the whole
correction by `raw_t >= replace_threshold`. At the default, since `raw_t`
never goes negative, the correction is **active on every single step** —
this is a hard constraint enforced throughout the whole trajectory, not an
occasional nudge. Raising it (e.g. to 0.5) relaxes the constraint once
`raw_t` drops below the threshold, letting late, low-noise refinement steps
move freely — the same "constrain early, relax late" pattern
`--sym_threshold` already uses for `--sym`'s post-hoc symmetriser (spec 06
§1.6).

Ported as `replace_to_blob_anchor` wired through `sample()`'s existing
`post_step_fn` hook (`salad_torch/diffusion/sampling.py`) — the same seam
`--sym`'s Hook C uses, no new hook needed in `sampling.py`. Mechanically this
is host-side, sequential, operating on `data["pos"]` **after** the model call
returns, so it composes with `--fast`'s compiled/CUDA-graph path exactly as
`--sym`'s Hook C would if `--fast` allowed `--sym` (it doesn't, because
Hook B mutates `pos_noised` *inside* the graph's captured region — `--blobs`
needs no equivalent of Hook B, so `--blobs` + `--fast` is not similarly
blocked).

## 3. Precedence / incompatibility decisions (port-side scope)

Upstream's `eval_ve_shape.py` CLI surface (`parse_options` defaults) is:
`params, out_path, blobs, blob_scale, config, timescale_pos, timescale_seq,
num_aa, sym, merge_chains, sequence_only, renoise, sample_final,
replicate_count, screw_angle, screw_translation, screw_radius, sym_mean,
num_designs, num_steps, out_steps, dssp_mean, dssp, template, template_aa,
prev_threshold, replace_threshold, cloud_std, depth_adjust, sym_threshold,
jax_seed`. Several of these are declared but **never read** in the
generation loop body — the same "config.symm-style dead branch" pattern
spec 06 §1.2 documents for `--sym` in the de-novo script:

* `--sym`, `--sym_mean`, `--sym_threshold`, `--replicate_count`,
  `--screw_angle`, `--screw_translation`, `--screw_radius`: declared, never
  read. `eval_ve_shape.py`'s `__main__` has no symmetriser call site at all.
* `--template`, `--template_aa`: **not even declared** in this script's
  `parse_options` call (unlike `--dssp`/`--dssp_mean`, which are both
  declared *and* used).

`apply_blob_chain` (`scripts/generate.py`) therefore rejects, with
`SystemExit`, combining `--blobs` with: `--template`, `--order_agnostic`
(also absent from upstream's flag surface entirely), `--sym`, `--group_size`,
`--batch_size` (the port's own grouped path structurally cannot carry a
`post_step_fn` — see `sample_group`'s docstring — so it cannot carry §2's
hard-replace mechanism either), and `--dssp` (literal string or `"random"`).

The `--dssp` case is the one genuine judgment call: upstream *does* wire
`--dssp`/`--dssp_mean` for this script (both declared and read), but its own
`--dssp` handling for this script is **simpler** than the general precedence
chain's step 3 (`notes/specs/06_generation_flags.md` §0) — it only
reassigns `num_aa`, never touches `resi`/`chain`/the point cloud:

```python
if dssp is not None and not isinstance(dssp, Callable):
    num_aa = dssp.shape[0]
```

Combined with a blob CSV of a different length, this silently orphans
`init_pos`/`resi`/`chain` at the old (blob) size while `num_aa` moves to the
new (dssp) size — a shape-mismatch crash a few calls downstream, upstream's
actual (untested — its own example command never combines `--dssp` with
`--blobs`) behaviour. The port's general `apply_precedence_chain` dssp
branch is *differently* built (rebuilds `resi`/`chain`/`base_chain` too),
so reusing it here would not reproduce upstream's specific crash mode
either. Rather than pick one of two non-identical, both-untested branches
and call it "the" faithful reproduction, `apply_blob_chain` rejects the
combination outright with a clear error. `--dssp_mean` has no such
ambiguity — it is a pure per-residue broadcast (`prepare_condition` expands
a `(3,)` request over the whole flat axis) independent of `num_aa`/`resi`/
`chain`/the point cloud, and upstream wires it unconditionally alongside
blobs; the port does too. `--merge_chains` also composes normally — it is a
no-op either way, since `parse_blobs`' `chain` is already all-zero.

Not restricted (compose normally, generic CLI/inference-speed machinery
orthogonal to shape conditioning): `--fast`/`--preset` (§2's compatibility
note above), `--compile`, `--depth_adjust`, `--seed`/`--jax_seed`,
`--sequence_only`, `--sample_final`(`_temperature`), `--renoise`,
`--timing_json`, `--cloud_std`.

## 4. What was ported, file by file

* `salad_torch/inference/blobs.py` (new) — `parse_blobs` (§0),
  `replace_to_blob_anchor` (§2). Pure CLI/setup-level composition, same
  restriction as the rest of `salad_torch/inference/` (spec 06's package
  docstring): no changes to `salad_torch/modules/*`.
* `scripts/generate.py` — `--blobs`/`--blob_scale`/`--replace_threshold`
  CLI flags; `apply_blob_chain` (§3, mirrors `apply_precedence_chain`'s
  return contract plus `blob_init_pos`); a 5-line dispatch in `main()`
  (`pc = apply_blob_chain(...) if args.blobs else apply_precedence_chain(...)`);
  one `data["pos"] = blob_init_pos.to(device)` override per design, right
  after `make_design_data` (which still builds a correctly-all-zero
  `init_prev["pos"]`, since it is `zeros_like`'d from the pre-override
  cloud — matching upstream's own `init_prev = dict(pos=jnp.zeros_like(init_pos), ...)`,
  also all-zero regardless of what `init_pos` holds); `post_step_fn`
  extended to also call `replace_to_blob_anchor` when blobs are active
  (mutually exclusive with `--sym` per §3, so the two branches' relative
  order never matters for any reachable command line).

## 5. Evidence this actually works (not just "does not crash")

`.agent-research/results/N034-shape-conditioning/` — per-blob CA centroid
distances (generated vs. requested) and per-residue nearest-blob-assignment
accuracy, on GPU, both eager and (where compatible) with a real blob CSV.
See `RESULTS.md` there for the numbers.
