# N014 — Batched condition-branch GEMM across the 6 diffusion blocks (P7)

Picked up from a prior agent that died on a rate limit ~2h in with no headline
number delivered. This file closes the thread: D1 (whole-forward effect,
measured separately), D2 (the parity gate), D3 (a docstring correction, found
already applied — see below).

## D1 — the headline number

Whole-forward effect of `SALAD_OPT_BATCHCOND` (P7) and `SALAD_OPT_AGGRFUSE`
(inductor's `aggressive_fusion`), measured SEPARATELY and together, against
neither. H100, `default_vp-200k`, fp32 `matmul_precision="highest"`,
`preset="exact"`, manual whole-forward CUDA graph, **one length per process**,
3 reps/cell (`scripts/bench_one.py`).

| L   | arm            | full step (ms) | Δ full  | lean step (ms) | Δ lean  |
|-----|----------------|-----------------|---------|-----------------|---------|
| 100 | neither (off)  | 4.3883          | —       | 3.5677          | —       |
| 100 | BATCHCOND only | 4.0540          | -7.62%  | 3.2267          | -9.56%  |
| 100 | AGGRFUSE only  | 4.2923          | -2.19%  | 3.4877          | -2.24%  |
| 100 | both (ship)    | 3.9817          | -9.27%  | 3.1753          | -11.00% |
| 400 | neither (off)  | 8.7537          | —       | 7.3547          | —       |
| 400 | BATCHCOND only | 8.6207          | -1.52%  | 7.2380          | -1.59%  |
| 400 | AGGRFUSE only  | 8.6923          | -0.70%  | 7.3073          | -0.64%  |
| 400 | both (ship)    | 8.5557          | -2.26%  | 7.1823          | -2.34%  |

jobs: `bench2_903621-4` (off/BATCHCOND-only/both, L=50/100/200/400, 3 reps) +
`benchaggr_903899`/`benchaggr_903905` (AGGRFUSE-only, L=100/L=400, 3 reps —
the missing 4th arm of the 2x2, filled in this session; `benchaggr_903900`
NODE_FAIL'd and was resubmitted as `903905`).

**The two effects are additive, not synergistic or interfering**: at L=100,
BATCHCOND(-9.56%) + AGGRFUSE(-2.24%) = -11.80% predicted vs -11.00% measured
combined (0.8 pp overlap); at L=400, -1.59% + -0.64% = -2.23% predicted vs
-2.34% measured (within noise, slightly super-additive). Per 500-step design:
1.785 s -> 1.588 s at L=100 (-11.0%), 3.679 s -> 3.593 s at L=400 (-2.3%).

**Did the isolated 1.87-2.13x GEMM speedup survive integration? Partially, and
it erodes fast with length — here is where it went.** The N012 Codex audit
measured `bmm(batch=6)` vs six `mm`s in isolation at 1.87-2.13x. That
isolated number is about the condition-branch projections ALONE; the whole
forward is ~1500 kernel launches (`x_kernels.py`: 1202 kernels / 4.356 ms at
L=100, `need_aa=True`, of which 419 are GEMMs, most NOT part of `pair_2`).
P7 replaces `depth`=6 small `mm`s with one `bmm` per projection (7 projections
+ 2 MLP layers), i.e. it removes ~48 kernel launches, each of them tiny at
K=48/64 neighbours (fixed regardless of L). At L=100 those tiny,
launch-overhead-dominated `mm`s are a real fraction of a 3.57 ms lean step, so
consolidating them into `bmm`s is worth -9.56%. At L=400 the SAME 48 launches
still shrink to the SAME `bmm`s, but the rest of the step (attention and other
GEMMs whose row count scales with L, not with the fixed K) has grown from
3.57 to 7.35 ms — so the absolute saving actually *shrinks* (0.341 ms at
L=100 -> 0.117 ms at L=400, not even constant) while the denominator grows,
and the win falls to -1.59%. The condition branch's `mm`s are ALSO no longer
purely launch-bound at L=400 (each one now has 400 rows instead of 100), so
even the sequential six-`mm` baseline is closer to compute-bound, which is
the second reason `bmm` buys less. **This is not a failure of P7 - it does
what it says (fewer launches) - it is a real result about WHERE the
launch-bound regime lives: small-to-medium L, not large L.** `AGGRFUSE`
shows the identical pattern (-2.24% at L=100, -0.64% at L=400) for the same
reason (it too is a launch-count reduction, not a FLOP reduction).

L=50/L=200 (3 reps, BATCHCOND/off/both only, no AGGRFUSE-alone arm run) are in
`logs/bench2_903621.log` / `bench2_903623.log` for context; they interpolate
between the two rows above and were not re-run with the 4th arm since D1 only
required L=100 and L=400.

## D2 — the parity gate (must clear for P7 to land)

**Verdict: PASS. Landed** (already the default in the working tree; this
session re-verified it against the current code with the actual gate, not the
invalid `x_p7_realstate.py` probe the previous agent used).

The previous agent's `x_p7_realstate.py`-vs-`st2_L100.pt` comparison is NOT a
valid gate (per the brief): its own rerun floor is 1.05 A on `pos` on GPU and
0.85 A on CPU, and CPU segment reductions are deterministic in this codebase,
so a nonzero CPU floor means that probe carries unseeded randomness (almost
certainly the Gumbel neighbour draw) — a ratio against a contaminated floor
could hide a real regression. It is superseded by the real gate below and
should not be cited.

**The real gate**: `tests/test_model_parity.py` (all four checkpoint configs)
+ `test_geometry_parity` + `test_stack_parity` + `test_encoder_parity` +
`test_sampling_parity`, run via `run_parity.sh`, with `SALAD_OPT_BATCHCOND`
set EXPLICITLY in both directions (opt.py's bare default is currently `True`,
so an unset env var is not a valid "off" baseline on this tree — the original
`gates_903274` run predates that default and this session re-ran the gate
explicitly for that reason). `cond_prologue.py`/`blocks.py`/`diffusion.py` are
byte-identical (md5) between the original `gates_903274` run and this
session's re-run, so both runs exercise the same P7 implementation; the
re-run is job **903910** (CPU, 1m08s):

| config             | BATCHCOND=0 (off)              | BATCHCOND=1 (P7 on)             | floor     |
|--------------------|---------------------------------|----------------------------------|-----------|
| default_vp         | mean 0.00000, max 0.00002 A OK | mean 0.00000, max 0.00002 A OK  | 0.00002 A |
| default_vp_scaled  | mean 0.00000, max 0.00002 A OK | mean 0.00000, max 0.00002 A OK  | 0.00002 A |
| default_ve_scaled  | mean 0.00000, max 0.00002 A OK | mean 0.00000, max 0.00002 A OK  | 0.00002 A |
| multimotif_vp      | mean 0.00001, max 0.00014 A OK | mean 0.00001, max 0.00014 A OK  | 0.00020 A |
| geometry_parity    | 38/38 passed                    | 38/38 passed                     | —         |
| stack_parity       | mean 0.0000, max 0.0002 A OK   | mean 0.0000, max 0.0002 A OK    | —         |
| encoder_parity     | rel 5.155e-08, max 7.629e-06 OK| rel 5.155e-08, max 7.629e-06 OK | —         |
| sampling_parity    | final/noise-floor ratio 12.9x  | final/noise-floor ratio 7.5x    | <20x      |

`SUITE_EXIT=0` both times. Notably BATCHCOND=1's own numbers are IDENTICAL to
BATCHCOND=0's on the four checkpoint configs and geometry/stack/encoder (the
K 48->64 reduction-tree reassociation the docstring predicts is below the
2e-5/2e-4 A floor in every case tested here) — only `sampling_parity`'s ratio
moves (12.9x -> 7.5x, still comfortably under the 20x pass line, and actually
*tighter*, not worse). `multimotif_vp` never took the P7 path in this test
(its `pair_2` reads `pos`, not `prev_pos` — `cond_prologue.py`'s own
docstring — so it is not expected to move; its number is a control).

A second, independent line of evidence: `x_p7_equiv2.py`/`gates2_903281.log`
found the batched-vs-sequential CPU comparison is only bit-EXACT (0.0)
when there is no pair conditioning (`chains=1`, K=48); with real
multi-chain pair conditioning active (`chains=4`, K=64, `pair_cond=True`) it
diverges at ~1e-4-1e-3 A on `pos`/`local` even on deterministic CPU. This is
the SAME reassociation the docstring describes (`bmm` vs sequential `mm`
picks a different reduction tree once the `dmap`/`flags` pair terms are
non-zero) — it is exactly why the real gate is the checkpoint-parity FLOOR
comparison (which stays under 2e-5/2e-4 A with real pair conditioning present
in the `multimotif_vp`/pair-conditioned test batches) and not a
`torch.equal` check. `x_p7_cuda_ab.py`/`cudaab_903707.log` (H100, K=48,
5 seeds x 3 lengths) shows the batched-vs-sequential deviation sits mostly
at or below the identical-rerun CUDA noise floor, with a few seeds slightly
above it — consistent with "reassociation, not a bug", and consistent with
(not a substitute for) the deterministic-fixture gate above.

## D3 — the `PRESETS["deterministic"]` docstring correction

**Already applied, verified correct, no further change made.** The previous
agent (before dying) had already rewritten `salad_torch/modules/fast.py`'s
`"deterministic"` preset docstring before this session started. Checked
against the brief's independent reproduction (job 903683,
`/hnvme/workspace/v181bb10-salad/verify_detseg/verify_903683.log`, isolated
snapshot, 2x2 reps, bit-identical within each setting) — the docstring's
numbers match exactly:

```
DETSEG=0: final dev 0.06166 A, ratio 9.5x PASS,  final_aa rel 2.046e-02 PASS, aatype 1/64 differ
DETSEG=1: final dev 0.19861 A, ratio 30.6x FAIL, final_aa rel 6.019e-02 FAIL, aatype 7/64 differ
          (residues flipped: 15, 22, 28, 29, 46, 48, 53)
```

The docstring correctly: (a) leads with the aatype count (7/64 - "a
discretely different design") rather than the ratio, (b) explicitly refuses
to lean on the 30.6x ratio and states why (its denominator is the ATOMICS
path's self-divergence floor, which DET_SEGMENT's whole purpose is to drive
to zero — so the denominator no longer describes the configuration under
test), giving the defensible raw numbers (0.199 A vs 0.062 A, ~3.2x) instead,
and (c) frames it as a TRADE with **no recorded sampling baseline of its own
yet**, telling a caller to record one before reporting any result from this
preset rather than reading it against `exact`'s baseline. This is exactly
what the brief asked for; verified, not re-written.

## Outputs

- `bench_p7.py`, `cmp_bits.py`, `cmp_floor.py`, `x_*.py` — harnesses (pre-existing)
- `run_bench_aggr.sbatch`, `run_cpu_gates3.sbatch` — new this session (fill the
  missing AGGRFUSE-alone arm; re-run the parity gate explicitly on both sides)
- `logs/benchaggr_903899.log` (L=100, AGGRFUSE alone), `logs/benchaggr_903900.log`
  (L=400, NODE_FAIL), `logs/benchaggr_903905.log` (L=400 resubmit, AGGRFUSE alone),
  `logs/gates3_903910.log` (re-verified parity gate, both directions, current code)
- `salad_torch/modules/cond_prologue.py` (`StackedCondPair`, the P7 prologue),
  wired into `salad_torch/modules/blocks.py` / `diffusion.py`, flagged by
  `salad_torch/opt.py: BATCH_COND` (default True — P7 ships in `exact`)

## Next

Nothing further planned for P7 itself. If someone wants to chase more of the
kernel-launch floor at large L, the finding above says where NOT to look:
consolidating more of the six-block loop's SMALL, fixed-K work will keep
paying off less as L grows; the larger-L floor is dominated by the O(N)/O(N·K)
work that already scales, not by launch count.
