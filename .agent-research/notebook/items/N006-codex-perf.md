# N006 · Codex performance gap investigation

type: fix · date: 2026-09-25 · status: open

## Goal
Measure and close the fp32 H100 L=100 compiled latency gap without changing numerical parity.

## Plan
1. Baseline: independent L=100 H100 CUDA event profile and wall timing via `profile_baseline.py`, Slurm job 902168, one GPU x 35 min (maximum 0.583 GPU-hour). GPU placement is necessary because the target is H100 latency; CPU jobs are not being submitted.
2. Candidate 1 hypothesis: profile showed 82 Torch-Compiled Region calls/forward and wall time 7.06 ms versus 5.37 ms summed CUDA kernel time. Capturing the entire default-mode compiled core in one explicit CUDA graph may cut the ~1.69 ms gap without changing arithmetic or neighbour selection. Baseline is job 902168; expected gain >=0.2 ms. One variable: graph capture strategy.
3. Test candidate from `results/N006-codex-perf/codex/` against all exact parity commands and one-length H100 benchmark.

## Changes

## Evidence
- Baseline job 902168, completed 00:04:38, 0:0, H100 at L=100: 7.0580 ms/50 and 7.0596 ms/100; 1,904 CUDA events/forward (837 Triton, 413 GEMM, 295 DtoD copies plus 359 other); 5.370 ms summed CUDA kernel time/forward. Source: `baseline_902168.log` lines 42-46.
- Baseline job 902168: `baseline.sbatch`, `profile_baseline.py`; output `baseline_902168.log`.

## Outcome

## Outputs
<!-- files in .agent-research/results/N006-slug/ + external absolute paths -->

## Next
