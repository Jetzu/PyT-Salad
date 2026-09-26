# N006 · Codex performance gap investigation

type: fix · date: 2026-09-25 · status: open

## Goal
Measure and close the fp32 H100 L=100 compiled latency gap without changing numerical parity.

## Plan
1. Baseline: independent L=100 H100 CUDA event profile and wall timing via `profile_baseline.py`, Slurm job 902168, one GPU x 35 min (maximum 0.583 GPU-hour). GPU placement is necessary because the target is H100 latency; CPU jobs are not being submitted.
2. Candidate 1 hypothesis: profile showed 82 Torch-Compiled Region calls/forward and wall time 7.06 ms versus 5.37 ms summed CUDA kernel time. Capturing the entire default-mode compiled core in one explicit CUDA graph may cut the ~1.69 ms gap without changing arithmetic or neighbour selection. Baseline is job 902168; expected gain >=0.2 ms. One variable: graph capture strategy.
3. Candidate 2 hypothesis: the condition-cache key performs GPU reductions and `.item()` host synchronizations each forward; replacing these with tensor identity plus version counters may reduce wall time by >=0.1 ms and also fix equal-sum cache collisions. Baseline is candidate 1 job 902218 at 4.9138 ms. One variable: cache-key strategy.
4. Test candidate from `results/N006-codex-perf/codex/` against all exact parity commands and one-length H100 benchmark.

## Changes

## Evidence
- Candidate 1 job 902218, completed 00:04:35, 0:0, H100 at L=100: 4.9144 ms/50 and 4.9138 ms/100; 1,483 CUDA events/forward; 4.861 ms summed GPU kernel time/forward. Source: `manual_graph_902218.log` lines 32-36.
- CPU parity job 902241, completed 0:0: all four model configs OK, geometry 38/38, stack OK, encoder OK. Sources: `parity_902241.log` and `parity_*.log`.
- New compiled/eager check job 902240 failed a 2e-4 A position threshold at 2.8126e-4 A; compare job 902292 found original reduce-overhead already 7.3814e-4 A vs eager and manual graph 1.1032e-3 A vs eager on a different random input. The additional 3.6626e-4 A needs attribution; all aatypes match. Sources: `graph_check_902240.log` line 32 and `compare_modes_902292.log` line 37.
- Baseline job 902168, completed 00:04:38, 0:0, H100 at L=100: 7.0580 ms/50 and 7.0596 ms/100; 1,904 CUDA events/forward (837 Triton, 413 GEMM, 295 DtoD copies plus 359 other); 5.370 ms summed CUDA kernel time/forward. Source: `baseline_902168.log` lines 42-46.
- Baseline job 902168: `baseline.sbatch`, `profile_baseline.py`; output `baseline_902168.log`.

## Outcome

## Outputs
<!-- files in .agent-research/results/N006-slug/ + external absolute paths -->

## Next
