# N025 · Rollout regression mechanism: where and what state

type: research · date: 2026-09-26 (finished 2026-09-27) · status: done

## Goal

Find WHERE IN THE 250-STEP ROLLOUT `--fast --preset fast` at fd9ba32 goes wrong
and WHAT STATE is responsible — independent of which commit introduced it.
c5ab7a0: CA-CA 3.7613, Rg 12.90, helix 0.580, 0/160 diverged. fd9ba32: helix
0.85-0.89, Rg 18.5-24.1, 26-41/160 diverged. Every parity test passes at
0.00002 A because every one of them is a SINGLE forward against a JAX fixture.

Method-diverse from the two Codex red-team agents (who audit COMMITS) and from
the 5-arm env-veto bisect already running as job 904678 (which counts ARMS).
This item measures the MECHANISM.

## Plan

1. **Eager vs compiled, first.** Same pinned schedule (250/200), same seed, at
   BOTH commits: `--fast --preset fast` / eager / `--fast --preset exact`. If
   eager is clean and `--fast` is broken, the defect is compilation/capture.
2. **Per-step trajectory.** Record CA-CA, Rg, max|pos|, ||prev_pos||,
   ||prev_local|| and content hashes of `prev_pos`, `pos_out`, `pos_noised` and
   the Gumbel `uniform` field at every one of the 250 steps, at both commits.
   Gradual vs sudden; first step over 5 A.
3. **Self-conditioning.** From the hashes: is `prev_pos` at step N the output of
   step N-1? Is the `prev_threshold` reset applied at the same steps?
4. **Randomness reuse.** Consecutive steps must draw DIFFERENT noise and a
   DIFFERENT Gumbel field. Plus the test a fixture cannot do: an
   INPUT-SENSITIVITY probe — hold everything bitwise fixed and change ONE input
   (each of the 19 `uniform` slices on its own, `prev_pos`, `prev_local`),
   measure |d(pos)| against the identical-call floor. A parity fixture pins
   `uniform`, so a dead random draw passes it.
5. **CPU bitwise commit diff.** CPU is deterministic by construction, so
   `forward_core` good-vs-HEAD on identical dumped inputs separates MODEL CODE
   from compilation with no noise-floor argument at all.
6. **CSE veto.** The N017/O1 `fx_graph_cse` pass has no env veto (it is a local
   closure in `FastDenoiser.__init__` at fd9ba32). Vetoed by making
   `torch._functorch.compile_utils.fx_graph_cse` the identity AFTER
   `torch._functorch.partitioners` has bound the real one — deleting it breaks
   the whole inductor import.

## Changes

None to the repo. All instrumentation lives outside the tree in
`/hnvme/workspace/v181bb10-salad/n025_x/`, driving two detached worktrees
(`n025` = fd9ba32, `n025_good` = c5ab7a0).

## Evidence

Jobs 906600 906623 906627 906633 906640 906656 (mine, 2026-09-27) plus the
never-read logs of 904716 904722 904739 (2026-09-26) and the caller's 906549 /
906583. Full tables in `../results/N025-regression-mechanism/RESULTS_PART2.md`;
logs and scripts alongside it in `scripts2/`.

**1. The regression is P7 (`603051e`), nothing else.**
`--fast --preset fast`, L=100, seed 42: c5ab7a0 0/64 (3.7618 / 13.06) ->
**603051e 11/64 = 17.2 % (3.8738 / 20.75)** -> b9dd416 11/64 (job 906583). At
HEAD, `SALAD_OPT_BATCHCOND=0` gives **0/160 (3.7632 / 12.94)**, i.e. the
reference numbers back exactly, and both-off 0/160 (3.7610 / 12.93) — while
`SALAD_OPT_AGGRFUSE=0` is still 33/160 (job 906549). CSE vetoed: still 5/32
(904716 arm D). Compiled without CUDA-graph capture: still broken (904722 L1).

**2. EAGER IS CLEAN; the compiled path is the broken one.** HEAD eager 0/64,
CA-CA 3.7614, Rg 12.98 +- 1.72 on the production schedule (906600 E1), and
0/32 / 3.7608 / 12.95 on the pinned 250-step schedule (904716 arm B) vs
c5ab7a0 eager 3.7601 / 12.77. HEAD compiled is broken in BOTH presets
(fast 12.5 %, exact 21.9 %). CAVEAT: this alone does NOT exonerate P7's maths —
`opt.BATCH_COND` defaults True from the env and is only WRITTEN by
`FastDenoiser.__init__`, so eager runs P7 too, but through `diffusion.py`'s eager
branch, which feeds the prologue six IDENTICAL neighbour index sets.

**3. P7's MATHS IS EXACT.** At a dumped step-30-of-250 state (`prev_pos|max|`
24.48, K=48, per-block neighbour sets verified distinct), cache `in_sync=True`:
prologue vs six sequential `pair_2` = **0.0 to 8.6e-6 abs, <=6.2e-7 rel**, pair
MASKS bit-identical in all 6 blocks, GPU and CPU. `StackedCondPair.forward`
compiled ON ITS OWN is also correct (8.6e-6). Control: sequential compiled vs
sequential eager is the SAME 8.6e-6.

**4. INDUCTOR MIS-COMPILES THE PROLOGUE INSIDE `forward_core`.** One forward,
identical inputs, `DET_SEGMENT=1`, fp32, plain `torch.compile(mode="default")`:

| comparison | pos max abs | local | aatype |
|---|---|---|---|
| EAGER P7 on vs off | 9.60e-06 | 6.1e-04 | 0 |
| **COMPILED P7 on vs off** | **9.586 A** | **646** | **12** |
| P7 OFF compiled vs eager (FLOOR) | **2.00e-05** | 6.1e-04 | 0 |
| **P7 ON compiled vs eager** | **9.586 A** | **646** | **12** |

Five orders of magnitude above the floor measured on the same forward. Per
block: [0.368 4.16 5.95 5.51 8.58 9.59] — already wrong at block 0's output.

**5. It is an inductor FUSION bug.** All COMPILED, P7 ON, vs EAGER P7 ON:
`aggressive_fusion=False`, `pattern_matcher=False`, `epilogue_fusion=False`,
`inline_inbuilt_nn_modules=True` and `mode="reduce-overhead"` every one returns
the BIT-IDENTICAL wrong 9.586 A; **`max_fusion_size=1` returns 2.77e-05, i.e.
correct**, and `max_fusion_size=2` is already wrong again. So a single two-node
fusion in the P7 graph is enough. torch 2.5.1+cu121, sm_90.

**6. Deterministic, and wrong from step 0.** `SALAD_OPT_DETSEG=1` does not help
(P7 on 6/32 = 18.75 %, 3.8788 / 20.16; P7 off 0/32, 3.7643 / 13.12, job 906623
arms Q). Paired rollouts under DETSEG: P7-on vs P7-off differ by **9.74 A at
step 0** (where `prev_pos` is still the sampler's zeroed `init_prev`), 90-96 % of
`distogram_neighbours` disagree from step 1, and they end 40-62 A apart — while
a P7-off replicate is **bit-identical for all 250 steps, 3/3 designs** (906633).

**7. Two hypotheses REFUTED.** Randomness reuse: 0 duplicate `uniform` hashes and
0 duplicate noise hashes over 3216 instrumented steps at both commits, and the
compiled region contains no RNG op at all (the Gumbel field is pre-drawn outside
and copied into a static buffer), so `fx_graph_cse` has nothing random to merge;
vetoing it changes nothing. Self-conditioning: `prev` hashed in and out gives
fed-back 800 / zero-reset 2416 / **MISMATCH 0**, identical in all three arms, so
`b9dd416`'s refactor did not change when the static buffers are written or which
steps the `prev_threshold` gate fires on.

**8. Where the rollout degrades (job 904722).** Gradual, from step ~32 of 250:
converged-only Rg 12.19 -> 15.15 at step 32, 22.24 at 64, peak 24.36 at 80,
settling at 20.17 (c5ab7a0 stays 12.2-12.9 throughout); `max|pos|` 36-44 vs
24-38. The 620-688 A monsters are a SEPARATE late event — first CA-CA > 5 A at
step 175/177 with a single-step jump of x2.85/x3.37.

## Outcome

**P7 (`603051e`'s batched condition-branch prologue, `SALAD_OPT_BATCHCOND`) is
the whole regression, and the cause is an inductor FUSION miscompile, not
`cond_prologue.py`'s arithmetic.** P7's maths is right to fp32 rounding eagerly
and compiled in isolation; compiled inside `forward_core` it is 9.586 A wrong on
a single forward against a 2.0e-05 A floor, deterministically, from step 0.
`max_fusion_size=1` is the only knob of six that fixes it.

Invisible to the suite because all four parity tests drive the EAGER path, where
P7 is correct — and where `diffusion.py`'s eager branch additionally feeds the
prologue six identical neighbour index sets, so the fast path's distinct-index
case has never been tested at all.

Decision deferred to the user (diagnose-only task). The measured switch is
`SALAD_OPT_BATCHCOND=0` / dropping `batch_cond_pair` from the presets; N014
recorded P7's win as -9.27 %/-11.00 % whole-forward at L=100, eroding to
-2.26 % at L=400.

## Outputs

`.agent-research/results/N025-regression-mechanism/` — `RESULTS_PART2.md`,
`scripts2/` (x_p7.py, x_p7u.py, x_p7b.py, x_pair.py, cmp_*.py, wrap.py, the five
sbatch files), the seven job logs, `traj_analysis_904722.txt`. Per-step
trajectory JSONs (>5 MB) stay at `/hnvme/workspace/v181bb10-salad/n025_x/traj/`.

## Next

See TODO T8-T10.
