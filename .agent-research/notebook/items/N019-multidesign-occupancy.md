# N019 · Multi-design occupancy: N independent designs in one CUDA graph

type: task · date: 2026-09-26 · status: done

## Goal
N016 §7 measured that 8 independent copies of the compiled forward, captured
in ONE cuda graph on 8 streams, cost 3.6x one forward, not 8x -> 2.22x
(`exact`) / 2.26x (`fast`) more designs per second at L=100, with byte-
identical computation per design. N016's own caveat R4: that was measured on
`Diffusion.forward_core`, NOT on `scripts/generate.py`. Verify it end to end
on the production path, find the optimal N (and how it moves with L), ship it,
and gate it on every design being the design it would have been alone.

Added mid-task by the user: this must run on CONSUMER GPUs including Blackwell
sm_120. So N is a runtime parameter auto-selected ON THE TARGET DEVICE, peak
VRAM is reported per N, everything is portable PyTorch, and the path degrades
to the existing single-design one rather than failing.

## Plan
- `GroupFastDenoiser` (fast.py) + `sample_group` (sampling.py) +
  `--group_size` (generate.py). Members share ONE compiled forward and one
  captured graph; each keeps its own static buffers, its own noise and its own
  seed-derived RNG stream.
- GATE: design i of a group of N == design i of a group of 1, measured under
  `SALAD_OPT_DETSEG=1` (reproducible by construction -> require BIT-IDENTITY)
  and under the shipping atomics path against a same-job g1-vs-g1 floor.
- Curves: N in {1,2,3,4,6,8,12,16,auto} x L in {50..400} x {exact,fast}, ONE
  (preset, L, N) per PROCESS. Speed AND peak VRAM.
- Standard suite: test_model_parity x4 + geometry/stack/encoder/sampling.

## Changes
- `salad_torch/modules/fast.py`: `GroupFastDenoiser` (N members, N streams,
  one capture), `memory_bound_group`, measured `_autotune` for
  `--group_size auto`, `pin_condition`; `FastDenoiser.forward` split into
  `_prepare_inputs`/`_run`/`_postprocess` + `_copy_in` (behaviour-preserving,
  proven by the parity suite).
- `salad_torch/diffusion/sampling.py`: `sample_group`.
- `scripts/generate.py`: `--group_size`, `run_grouped`, `make_design_data`,
  `design_generator`.

## Evidence
| job | what | result |
|---|---|---|
| 904087/904088 | smoke | cancelled/failed - `mode="reduce-overhead"` inherited into the group's own capture ("Cannot prepare for replay during capturing stage"). Fixed: the head member forces `mode="default"` |
| 904093 | parity suite | SUITE_EXIT=0, all floors exactly as documented |
| 904098 | smoke, fixed | N=1/4/auto curve + DETSEG identity 4/4 bit-identical |
| 904122-904131 | N x L x preset curves + peak VRAM | 90 points, no failures |
| 904132/904133 | identity gate, full schedule, 8 designs | DETSEG **8/8 bit-identical, 0/800 aatype**, both presets |
| 904150-904153 | end to end on `generate.py` | fast L=100 0.603 -> 0.334 s/design |
| 904175 | final verification of the commit candidate | parity + gate + CLI surface |

## Outcome
**VERIFIED, at 1.81x not 2.26x.** `fast` L=100 end to end on the real
`scripts/generate.py`: **0.603 -> 0.334 s per design**. Decomposition of the
gap to N016: 2.26x (its core harness) -> 2.01x (production capture, replay
only; N016's k=1 denominator was inflated, its k=8 agrees to 0.5%) -> 1.81x
(per-design host work that does not overlap).

GATE PASSES in its strongest form: under DETSEG a grouped design is
BIT-IDENTICAL to the solo design, 8/8, both presets, full schedule, through
`generate.py`. Since DETSEG changes only the atomic ordering, that proves
grouping touches nothing else. The atomics arm CANNOT discriminate - its
self-divergence floor is already 48% of residues on a rerun of the same
command (an N009 property of this sampler, not of N019).

Curves: the win falls monotonically with L (exact best-N host-inclusive 2.61x
@L=50, 1.94x @L=100, 1.65x @150, 1.48x @200, 1.30x @300, 1.17x @400) and
saturates then slightly regresses past the knee. `--group_size auto`, which
MEASURES on the target device rather than evaluating a formula, landed on the
knee at all 10 (preset, L) points and always within 5% of the best pinned N.
Peak VRAM is linear at ~3.8 MiB/residue/design; on a 24 GB card memory never
binds at a group size worth using.

Scale-down: S ≈ 1 + 0.88/w (w = work per forward relative to L=100 on 132 SMs)
fits the whole L sweep to 0.14. Extrapolated by SM count: ~1.85x on a 4090,
~1.3x on a 46-SM part, ~1.0x below that. NOT measured - torch 2.5.1+cu121
cannot target sm_120 at all.

## Outputs
- `.agent-research/results/N019-multidesign-occupancy/` (RESULTS.md, curves.txt,
  x_curve.py, x_identity.py, cmp_identity.py, parse_curve.py, sbatch, logs)

## Next
- The latency half of N016's E1 (multi-stream capture of ONE forward's own
  independent work) is still untried; this item took only the throughput half.
- `--group_size` + `--renoise`/`--sequence_only` are refused, not impossible;
  both could be grouped if anyone wants them.
