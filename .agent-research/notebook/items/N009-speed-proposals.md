# N009 · Implement the ranked speed proposals (N007 PLAN P2-P8)

type: task · date: 2026-09-26 · status: done · commit: bd38a0a

## Goal
Take one denoiser forward from the measured 4.95 ms (L=100, manual whole-forward
CUDA graph, fp32 highest) as close to 2.0 ms as is achievable **without moving
the 0.00002 A parity floor**. Unit of account: KERNELS REMOVED.

## Plan
Execute N007 PLAN §4 in ranked order, one proposal at a time, re-measuring the
whole forward after each: P1 (likely subsumed by the manual graph), P2, P3, P4,
P5, P6, P7, and P8 only if budget remains.
Gate after every change:
  SALAD_CONFIG={default_vp,default_vp_scaled,default_ve_scaled,multimotif_vp} tests/test_model_parity.py
  tests/test_geometry_parity.py (38/38), tests/test_stack_parity.py, tests/test_encoder_parity.py

## Changes
- `salad_torch/opt.py` (new): per-proposal switches, read once at import so they
  are Dynamo constants. Default ON; the env vars exist only so one job can
  measure each proposal in its own process.
- **P0 (not in the plan, found here)** `DotDict.__getattr__ = dict.get` is
  untraceable, so EVERY `config.<key>` read inside `forward_core` broke the
  Dynamo graph: `torch._dynamo.explain` reported **30 graphs / 29 breaks**.
  Hoisted every hot config value into plain attributes in `Diffusion.__init__`
  and `DiffusionBlock.__init__`, inlined `preconditioning_scale_factors`'s two
  config reads, and replaced the per-call `_slice(uniform, i)` helper (which
  blew Dynamo's cache_size_limit of 8 with 19 draws) by one `torch.unbind` up
  front -> **1 graph / 0 breaks**.
- P3 `HkLinear.forward_indexed` + all 6 one-hot@W call sites (relpos x3 module
  types, aa x2).
- P6 `extract_neighbours(index_neighbours=)`, `get_spatial_neighbours(distance=)`,
  `extract_condition_neighbours` split into `condition_neighbour_prologue` /
  `_tail`; hoisted in both `forward_core` and the eager `forward` (so the
  parity tests cover it).
- P4+P5 `forward_core(pair_cond=False)` -> `pair_condition=None` into the pair
  blocks and `num_block=0` in the condition selector (K 64->48).
  `FastDenoiser` derives the gate from the real `dmap_mask`/`flags`.
- P2 `forward_core(need_aa=False)` returns `(pos, local, trajectory)` only;
  `SamplerConfig.lean_steps` + `FastDenoiser(need_aa=)`; `FastDenoiser` keeps
  one captured CUDA graph per (need_aa, pair_cond).
- `FastDenoiser(preset="exact")` (default) / `scripts/generate.py --preset`.
- BUG FIX: `generate.py --fast` never threaded the seeded generator into the
  Gumbel neighbour draw, so `--seed` did not pin a fast run.

## Evidence
Baseline parity (before any change, 2026-09-26, login node CPU):
  default_vp 0.00002 A / default_vp_scaled 0.00002 A / default_ve_scaled 0.00002 A
  multimotif_vp 0.00020 A · geometry 38/38 · stack max 0.0003 A · encoder max 0.00001 A
Baseline speed (from N007/N008): 4.95 / 4.95 / 4.96 ms at L=100, 4.22 ms at L=50.
Parity after EVERY change (re-run each time, unchanged throughout):
  default_vp 0.00002 / default_vp_scaled 0.00002 / default_ve_scaled 0.00002 /
  multimotif_vp 0.00020 A; geometry 38/38; stack 0.0003 A; encoder 0.00001 A;
  sampling-parity ratio 8.3x the self-divergence floor, 1/64 aatype (unchanged).

### The reproducibility control failed - read this before any PDB diff
`e2e.sbatch` (job 903106) ran the SAME command, same `--seed 1234`, twice
(`allon` vs `allon2`): **the written PDBs DIFFER**. So `--fast` generation on
CUDA is not reproducible at all, independently of anything in N009, and
"the written structure is unchanged" cannot be used as a gate for any
optimisation. Suspected source (job 903136): `get_neighbours` chains stages
with `distance[index[:, None], neighbours] = update`, and a row holding the -1
sentinel writes to column N-1 twice with different values (notes/04 section 4,
bug B1). `index_put_` with duplicate indices is documented non-deterministic on
CUDA. The usable gate is the SINGLE FORWARD against a noise-floor control,
which is what `x_equiv.py` does.

### Ladder, L=100, one forward (full = reporting step, lean = the other 499)
| rung | full ms | lean ms | job |
|---|---|---|---|
| N007 baseline (manual CUDA graph, 30 Dynamo graphs) | 4.966 / 4.941 | 4.489 / 4.485 | 903079 |
| + P0 graph-break fix | 4.886 / 4.870 | 4.006 / 4.015 | 903105 |
| + P3 one-hot -> gather | 4.713 | 3.906 | 903105 |
| + P6 hoist neighbour stages | 4.650 | 3.828 | 903105 |
| + P4+P5 skip dead pair conditioning, K 64->48 | **4.385** | **3.564** | 903126 |
| (control) need_aux=0 | 4.378 | 3.572 | 903126 |
End to end `generate.py --num_aa 100 --fast`: **1.5 s/design** (was ~2.1 s).

## Outcome
4.95 -> **4.39 ms** on a reporting step and **3.56 ms** on the 400 steps that
do not report, at the unchanged parity floor. 2.0 ms is NOT reachable on the
exact path: see `README.md` / notes/04 section 7 for the kernel breakdown.
Two findings beyond the plan: the `DotDict` graph-break (P0) and the CUDA
nondeterminism that voids every trajectory-level test (`P2_NOTE.md`).

## Outputs
- `.agent-research/results/N009-speed-proposals/` - README, `P2_NOTE.md`,
  `x_equiv.py`, `x_nondet.py`, `x_kernels.py`, the sbatch ladders, `logs/`
- `notes/04_PARITY_AND_PERF.md` section 7
- jobs 903079, 903105, 903106, 903126, 903136, 903138 (h100, <1 GPU-h each)

## Next
- P7 (batch the condition branch across the 6 blocks) is ~0.2-0.4 ms for 2-3
  days and risks the parity budget on a batched GEMM - not taken on the exact
  path. P8 (a Triton pair-feature kernel) is the only thing that reaches 2.0 ms
  and is explicitly "materially different" numerics, so it belongs to `fast`.
- `--seed` does not pin a CUDA `--fast` run (`index_mean` -> scatter_add).
  Deserves its own item; fixing it changes summation order, so it needs a
  parity re-baseline.
- The conditioning cache still costs host syncs per step. An identity +
  `_version` key (holding a reference to the keyed tensors so ids cannot be
  recycled) would be exact and sync-free, and would remove the need for the
  `fast` preset `trust_cond_cache` shortcut.
