# N007 · Restructuring the SALAD diffusion block to match JAX speed

type: research · date: 2026-09-25 · status: backfilled (done; research plan, no production code)

Backfilled by work-lint 2026-09-26: `results/N007-block-restructure/` existed
with no INDEX row or item file. Reconstructed from `PLAN.md` and the logs
alongside it; no notebook entry was ever written at the time.

## Goal
Close the H100 speed gap (torch 6.5-7.0 ms vs JAX 4.73 ms fp32 at L=100)
without touching the released checkpoints or the 0.00002 A parity floor.
Research/measurement only - explicitly no production code was to be written.

## Plan
Measure where the forward's time actually goes (component isolation +
kernel census), rank candidate restructurings by kernels-removed (the real
bottleneck, not FLOPs), and hand a sequenced, gated plan to the implementers
(N009, N014).

## Evidence
All on H100, fp32 `matmul_precision=highest`, `torch.compile(mode=
"reduce-overhead")`, CUDA graphs, `default_vp-200k`, L=100 unless stated;
scripts `x_micro.py`, `x_pairsplit.py`, `x_batch6.py`, `x_e2e.py`, raw logs in
`logs/` (`run*.sbatch`, 1 GPU each, <=55 min, ~0.9 GPU-h/job).
- Measured GPU-busy 5.375 ms in ~1840 kernel launches (mean 2.9 us/launch);
  wall time 6.47-7.03 ms, so ~1.1-1.6 ms is launch gap, not GPU work.
  Arithmetic floor 0.21 ms (14.1 GFLOP @ 67 TFLOP/s fp32), memory floor
  ~0.37 ms - the model runs at ~13x its own hardware floor, bound by kernel
  count against a ~2-3 us/launch floor.
- Pair features x12 cost 2.03 ms (31%, 363 kernels/12=30.2 per block); of the
  7 per-block projections, only `dist` (in=400, ~51 TFLOP/s) and `pairvec`
  (in=240) are wide enough to matter - the other 5 (`relpos`, `dmap`, `flags`,
  `dir`, `rot`) each cost the same ~6.4 us as `dist` despite being tiny,
  because they are launch/latency-bound, not compute-bound. Dropping
  `relpos`/`dmap`/`flags` measured 1.307->1.051 ms (-12.6% of the pair stage).
- `torch.compile`/Inductor already CSEs the Gram-Schmidt frame construction
  across `direction_features` and `position_rotation_features` (built
  together: 0.719 ms; parts sum to 1.285 ms) - "hoist the frames by hand" is
  not a real proposal, it is already done.
- Fusion ideas in the original brief (pair-linear fusion, flash-IPA recast)
  were implemented and measured SLOWER than the current code - explicitly
  ruled out, not just deprioritized.
- TF32 (`matmul_precision="high"`) measures 6.47->5.94 ms (-8%) but reopens
  the N005-established ~6e-3 A coordinate drift; ruled off the table.
- Ranked proposals P1-P8 total an estimated -2.0 to -2.5 ms (-> ~4.2-4.6 ms,
  at/under JAX fp32); three are bit-exact, two reassociation-only, one (P7,
  batched condition-branch GEMM) touches the converter (identity transform -
  un-stacking what Haiku already stacked). P8 (custom kernel) is the only way
  to go materially below the ~2.5-3.0 ms launch-count floor after P1-P7, and
  is estimated at 1-2 weeks.

## Outcome
Research plan delivered, no code changed. Verdict: the gap is closable via
kernel-count reduction (not fusion/reassociation for FLOPs), landing at or
just under JAX fp32; going materially below needs either a custom kernel
(P8) or abandoning parity via reduced precision (ruled out). Standing rules
set for downstream implementers: never delete a parameter to skip a
computation, any derived weight is a non-persistent rebuilt buffer, classify
every change bit-exact/reassociation-only/materially-different up front,
re-measure the whole forward not the piece, fixtures stay CPU-generated (a
GPU JAX fixture is TF32 and not even self-reproducible, per N005).

## Outputs
`results/N007-block-restructure/`: `PLAN.md` (full writeup with the ranked
P1-P8 proposals and sequencing table), `common.py`, `x_micro.py`,
`x_pairsplit.py`, `x_batch6.py`, `x_e2e.py`, `run*.sbatch`, `logs/`.

## Next
Sequenced and handed to N009 (implemented the ranked proposals) and N014
(P7, batched condition-branch GEMM specifically). P8 (custom kernel) not
started.
