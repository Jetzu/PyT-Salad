# N014 · Batched condition-branch GEMM across the 6 diffusion blocks (P7)

type: task · date: 2026-09-26 · status: done

## Goal

Break the "3.4-3.5 ms exact floor" claimed in `notes/04_PARITY_AND_PERF.md` §7.8 by
landing N007 PLAN proposal **P7**: the six diffusion blocks' condition branch
(`DiffusionBlock.pair_2`) reads the loop-invariant `prev_pos`, so the six sets of
projections are one batched problem — stack the weights once into `(6, K, 64)` and
replace six `mm`s with one `bmm`.

The floor was refuted by the N012 Codex audit
(`.agent-research/results/N012-codex-floor-audit/FINDINGS.md`), which measured a
1.87-2.13x isolated speedup for `bmm(batch=6)` over six `mm`s at the shipping shapes.
Both that and the N007 six-block pair-builder experiment (1.017 -> 0.792 ms) are
ISOLATED. This item asks whether the gain survives integration.

## Plan

- implement `StackedCondPair` (the prologue) + wire it into `DiffusionStack`,
  `Diffusion.forward` and `Diffusion.forward_core`, behind `opt.BATCH_COND`
  (`SALAD_OPT_BATCHCOND=1`), default OFF until it clears the gates
- gates: four checkpoint parity tests at their floors, geometry/stack/encoder/sampling,
  CPU `torch.equal`, and the CUDA perturbation against the device's own
  run-to-run spread (>=30 reps)
- measure the WHOLE FORWARD at L=50/100/200/400, several repeats, one length per process
- K=48 (unconditioned) and K=64 (pair conditioning live) both measured

## Changes

- new `salad_torch/modules/cond_prologue.py`: `StackedCondPair`, the batched
  `pair_2` prologue over all `diffusion_depth` blocks (weights cached from the
  live per-block parameters, refreshed by a `load_state_dict` post-hook and by
  `sync()`; never used for `multi_motif` (`pair_2` reads `pos`, changes every
  block) or in `train()`/grad-enabled mode)
- `salad_torch/modules/blocks.py`: `DiffusionBlock.forward` takes an optional
  `cond_pair` (the prologue's per-block slice) and skips its own `pair_2` call
  when supplied; `DiffusionStack.batched_cond_pair_available()` gates it on
  `opt.BATCH_COND and cond_pair_stack is not None and not training and not
  grad_enabled`; `load_state_dict` post-hook refreshes the weight cache
- `salad_torch/modules/diffusion.py`: both `forward` (eager) and
  `forward_core` (fast/graph path) build the stacked condition-neighbour
  index up front and call the prologue once per forward when available -
  wired into BOTH paths so the four checkpoint parity tests (which drive the
  eager path) actually exercise it
- `salad_torch/opt.py`: `BATCH_COND` flag (`SALAD_OPT_BATCHCOND`, default
  **True** — see Outcome)
- `salad_torch/modules/fast.py`: `PRESETS["exact"]`/`["fast"]`/["deterministic"]
  all set `batch_cond_pair=True`; `FastDenoiser.__init__` applies it as
  `opt.BATCH_COND = bool(p.get("batch_cond_pair")) and _ENV_BATCH_COND` and
  refreshes the weight cache once per denoiser

## Evidence

**D1 (whole-forward effect, measured separately)** — H100, one length per
process, 3 reps/cell, `scripts/bench_one.py`:

| L   | neither | BATCHCOND alone | AGGRFUSE alone | both    |
|-----|---------|------------------|-----------------|---------|
| 100 lean | 3.5677 ms | 3.2267 ms (-9.56%) | 3.4877 ms (-2.24%) | 3.1753 ms (-11.00%) |
| 400 lean | 7.3547 ms | 7.2380 ms (-1.59%) | 7.3073 ms (-0.64%) | 7.1823 ms (-2.34%) |

jobs `bench2_903621-4` (off/BATCHCOND/both) + `benchaggr_903899`/`benchaggr_903905`
(AGGRFUSE alone, the missing 4th arm; `benchaggr_903900` NODE_FAIL'd, resubmitted).
Effects are additive (within ~1 pp) at both lengths. **The isolated 1.87-2.13x
GEMM speedup does NOT fully survive integration, and what survives erodes fast
with L**: it is worth -9.56% of a lean step at L=100 but only -1.59% at L=400,
because P7 removes a FIXED number of small, fixed-K launches while the rest of
the step (attention/other GEMMs scaling with L) grows around it — see the
N014 README for the full account of where the gain went.

**D2 (the parity gate)** — `test_model_parity.py` (all 4 checkpoint configs) +
geometry/stack/encoder/sampling, `SALAD_OPT_BATCHCOND` set explicitly in both
directions, current code (job 903910, CPU, `cond_prologue.py`/`blocks.py`/
`diffusion.py` md5-identical to the original `gates_903274` run that first
passed this): **SUITE_EXIT=0 both ways.** default_vp/_vp_scaled/_ve_scaled
stay at 0.00000-0.00002 A (floor 0.00002 A) with P7 on or off; multimotif_vp
0.00001/0.00014 A (floor 0.00020 A, unaffected — its `pair_2` never takes the
batched path); geometry 38/38; stack and encoder unmoved; sampling ratio
12.9x (off) -> 7.5x (on), both well under the 20x pass line (P7 is actually
*tighter*, not worse). Corroborating (not substitute) evidence:
`x_p7_equiv2.py` finds the batched-vs-sequential CPU comparison is bit-exact
(0.0) only when pair conditioning is inactive (K=48); with real multi-chain
pair conditioning (K=64) it reassociates at ~1e-4-1e-3 A even on
deterministic CPU — exactly why the checkpoint-parity FLOOR is the gate, not
`torch.equal`. `x_p7_cuda_ab.py` (H100, 5 seeds x 3 lengths) shows the
batched-vs-sequential deviation mostly at/below the identical-rerun CUDA
noise floor.

The previous agent's `x_p7_realstate.py` probe (comparing against
`st2_L100.pt` with a "rerun floor" argument) is explicitly NOT used as
evidence: its own floor is 1.05 A on GPU / 0.85 A on CPU, and a nonzero CPU
floor on a codebase whose CPU segment reductions are deterministic means the
probe has unseeded randomness in it (almost certainly the Gumbel neighbour
draw) — a ratio against a contaminated floor could hide a real regression.

**D3 (docstring correction)** — already applied by the previous agent before
this session started; verified against the brief's independent reproduction
(job 903683) and found to match exactly (0.199 A vs 0.062 A ~3.2x drift,
7/64 vs 1/64 aatype residues, correct refusal to lean on the 30.6x ratio,
correctly framed as a trade with no baseline of its own yet). No further
change made.

## Outcome

**P7 PASSES its gate. Landed in `exact`** (`opt.BATCH_COND` defaults `True`;
`PRESETS["exact"]["batch_cond_pair"] = True`). Whole-forward win: -9.27%
full / -11.00% lean step at L=100, shrinking to -2.26% / -2.34% at L=400 (see
D1). Combined with `aggressive_fusion` (N013), the shipped `exact` preset is
-9.27% / -11.00% at L=100 and -2.26% / -2.34% at L=400 vs neither optimisation.

## Outputs

`.agent-research/results/N014-batched-gemm/` — see its README.md for the full
D1 table, D2 gate table, and D3 confirmation.

## Next

Nothing further planned for P7 itself; see README "Next" for where NOT to
look for more launch-count wins at large L.
