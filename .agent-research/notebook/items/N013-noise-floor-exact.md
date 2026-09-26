# N013 · Re-examine the exact preset against the measured CUDA noise floor

type: research · date: 2026-09-26 · status: done

## Goal

N009 refused `aggressive_fusion`, `need_aux=False` and P7 against a 2.2e-4 A
CUDA run-to-run noise floor that was never directly measured on this
configuration. Measure the floor properly (frozen input, N identical
forwards, full distribution), then re-judge each refused optimisation against
it: is it bit-exact, reassociation-only-and-under-floor, or a real change?

## Plan

- `x_floor.py`: N=40 identical forwards on one frozen input (synthetic and a
  real mid-trajectory state), full pairwise max|d| distribution, on both CPU
  and CUDA
- `x_cpu_bits.py`/`cmp_bits.py`: CPU is deterministic by construction, so a
  process-global inductor flag (`aggressive_fusion`) can be checked for
  bit-exactness directly, no floor needed
- `x_detprobe.py`: does a deterministic segment reduction exist and what does
  it cost - if `index_sum`'s `scatter_add`/atomicAdd is the noise source, a
  one-hot-matmul reduction should be reproducible by construction
- gate whatever ships against the four checkpoint parity tests +
  geometry/stack/encoder/sampling, not against the floor number itself

## Changes

- `salad_torch/geometry/segment.py`: new `det_segment_sum` (one-hot matmul,
  deterministic by construction); `index_sum` takes it when `opt.DET_SEGMENT`
  and the data is floating-point
- `salad_torch/opt.py`: `DET_SEGMENT` flag (`SALAD_OPT_DETSEG`, default
  **False** - changes summation order, is its own parity baseline, never the
  default) and `AGGRESSIVE_FUSION` flag (`SALAD_OPT_AGGRFUSE`, default True -
  bit-exact, see Evidence)
- `salad_torch/modules/fast.py`: `PRESETS["exact"]`/`["fast"]` gain
  `aggressive_fusion=True`; new `PRESETS["deterministic"]` = `exact` +
  `det_segment=True`, with its docstring recording the sampling-parity
  finding below (already correct, verified not rewritten - see N014 D3)

## Evidence

**The floor**: ~1e-5 A on `pos` (mean 8.6e-06, p95 1.4e-05, max 2.6e-05 at
L=100, `floor2_903625.log`, 40 reps), not 2.2e-4 A. CPU is exactly 0.0.

**`aggressive_fusion`**: BIT-IDENTICAL, verified twice - CPU
(`cpubits_903241.log`, L=64/L=100, all 11 outputs, `WORST 0.000000e+00`) and
H100 under the deterministic segment reduction (`detgate_903291.log`,
`dg_synth_base` vs `dg_synth_aggr`, one shared bit pattern). Not a numerics
trade; -1.4 to -2.4% of a lean step at L=50/100 (isolated arm timing; see
N014's README for the D1 headline table that measures it alongside P7).

**`det_segment_sum`**: reproducible by construction (`--seed`: 40 identical
forwards -> 1 distinct bit pattern, `det_903261.log`/`detgate2_903317.log`),
+2.1% of the step. Does NOT hold `exact`'s sampling-parity floor - reproduced
independently this session (job 903683, isolated snapshot, 2x2 reps,
bit-identical within setting): DETSEG=1 final structure 0.19861 A (~3.2x the
DETSEG=0 drift from JAX), **7/64 residues take a different amino acid**
(discretely different design) vs 1/64 at DETSEG=0. The four checkpoint
configs + geometry/stack/encoder still hold at their usual floors; only
sampling moves. Shipped as its own preset (`"deterministic"`), never the
default, with this finding in its own docstring.

## Outcome

Both optimisations ship: `aggressive_fusion` in `exact` (bit-exact, no
caveat needed) and `det_segment_sum` as the separate `"deterministic"`
preset (a legitimate trade - reproducible CUDA runs, paid for in ~3.2x
JAX-drift and 7x the aatype divergence - NOT folded into `exact`, and
explicitly flagged as having no sampling-quality baseline of its own yet).
This re-judges N009's refusal: the refusal was against a floor 20x too
strict; both optimisations clear the REAL gate (checkpoint parity at its
measured floors), one bit-exact and one as an opt-in trade.

## Outputs

`.agent-research/results/N013-noise-floor-exact/` — see its README.md
"Results" section for the full account and job list.

## Next

Nothing further planned. `PRESETS["deterministic"]` has no recorded
sampling-quality baseline; record one before reporting results from it.
