# N023 · Red-team diagnosis: commit 603051e quality regression

type: research · date: 2026-09-26 · status: backfilled (done, superseded by N025)

Backfilled by work-lint 2026-09-27: `results/N023-regression-hunt/codex/` existed
with a full `FINDINGS.md` (382 files, 27 MB, no file >5 MB) but no INDEX row or
item file. This is a Codex red-team investigation, per the four-role workflow
(findings-only, no source edited); its conclusions were confirmed, extended,
and landed by N025.

## Goal
Diagnose the severe `--fast --preset fast` quality regression traced to commit
`603051e`, working from isolated `git archive` snapshots (`c5ab7a0`, `603051e`)
and targeted CPU/GPU probes.

## Plan
Test five candidate mechanisms in ranked order: stale stacked-weight cache,
`prev_pos`/motif `pos` split, first-step zero-`prev_pos` correctness,
`aggressive_fusion` bit-exactness, and second-call cache/graph-pointer
failure. Compare eager vs compiled vs captured-graph output at each stage.

## Evidence
Commit-level control: `c5ab7a0` gave 0/8 diverged, CA-CA 3.7602 A, Rg 12.59 A;
`603051e` gave 0/8 diverged (gate passed on the diverged-fraction threshold
alone) but Rg had already shifted to 19.92 A (job 904698). On the isolated
603051e snapshot, P7 on/off compiled-vs-eager step-0 position error was
9.7817 A vs 0.1418 A with the same input, and the full rollout Rg was 17.99 A
(on) vs 13.04 A (off) (jobs 904765/904772/904778). `aggressive_fusion` was
confirmed NOT bit-exact on the full compiled rollout (11.05 A at step 200,
job 904773) but did not reproduce the Rg quality shift in the n=8 A/B arms.
Stale-weight-cache and prev_pos/motif-split hypotheses were refuted or shown
inapplicable to the tested generator path.

## Outcome
Ranked P7's compiled batched condition-pair prologue (`SALAD_OPT_BATCHCOND`,
landed in 603051e) as the strongest suspected cause, with `aggressive_fusion`
non-bit-exactness as a secondary, non-explanatory numerical effect. Did not
isolate the exact faulty Inductor/CUDA kernel; recommended state-matched
eager/compiled/captured comparison as next step. No source file changed, no
commit/push/PR.

## Outputs
`results/N023-regression-hunt/codex/`: `FINDINGS.md` (full ranked diagnosis +
evidence limits), per-hypothesis `run_*.sbatch`/`*.log`, isolated `c5/`
and `603/` git-archive snapshot trees, `.npz` trace dumps (each <5 MB).

## Next
Superseded by N025, which confirms P7/`SALAD_OPT_BATCHCOND` as the WHOLE
regression (not just the strongest suspect), bisects the exact landing
commit, and localizes the mechanism to an inductor fusion miscompile
(`max_fusion_size=1` fixes it) rather than P7's own arithmetic. This item's
open question (exact faulty kernel) is the one N025 answers.
