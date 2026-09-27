# N024 · Red-team: is fx_graph_cse the regression, not P7?

type: research · date: 2026-09-26 · status: backfilled (done, superseded by N025)

Backfilled by work-lint 2026-09-27: `results/N024-regression-hunt-cse/codex/`
existed with a full `FINDINGS.md` (28 MB, no file >5 MB) but no INDEX row or
item file. This is a Codex red-team investigation, per the four-role
workflow (findings-only, no source edited); its conclusions were confirmed,
extended, and landed by N025.

## Goal
Follow-up to N023: test whether N017's `fx_graph_cse` post-pass (rather than
P7) explains the severe quality regression, via a controlled full-rollout
A/B/C/D arm comparison (CSE, P7 `BATCH_COND`, TF32 each toggled alone) plus
four CSE-specific sub-hypotheses (impure-node merging, random-draw collapse,
process-global pass leakage, stale buffer timing) and two more general ones
(stale conditioning pin, self-conditioning feedback break, grouping shift).

## Evidence
Controlled n=32 arms (`SALAD_OPT_DETSEG=1`, H100, `scripts/generate.py --fast
--preset fast`): production (CSE+P7+TF32 all on) 7/32 diverged, Rg 20.02 A;
CSE disabled 5/32 diverged, Rg 20.62 A; **P7 disabled 0/32 diverged, Rg
13.41 A** (recovers known-good geometry); TF32 disabled 7/32 diverged, Rg
20.46 A. The same pattern reproduced on an isolated pre-target `603051e`
snapshot: P7 off gives 0/32 diverged, Rg 13.44 A. `fx_graph_cse` was
confirmed to exclude RNG ops (`compile_utils.py`, direct `make_fx` probe) and
a real scope leak in `FastDenoiser.__init__`'s global `joint_custom_post_pass`
assignment was found but shown not to explain this CLI's failure. Stale
conditioning pin, self-conditioning feedback, and grouping-shift hypotheses
were each refuted or ruled inapplicable to the unset-`--group_size` path
tested.

## Outcome
**P7's `BATCH_COND` is the only tested toggle that restores normal geometry**;
CSE changes per-design trajectories (not bit-identical on/off under DETSEG)
but is not the severe-regression driver (both CSE arms still show invalid
designs at similar rates/Rg). Ranked P7 batched condition prologue as the
high-confidence cause, CSE as a non-primary secondary effect, and the
`b9dd416` legacy split refactor as very-low-likelihood. Proposed a
state-matched eager/compiled/captured comparison to isolate the exact P7
kernel fault. No source file changed, no commit/push/PR.

## Outputs
`results/N024-regression-hunt-cse/codex/`: `FINDINGS.md` (full ranked
diagnosis), `run_*.sbatch`/`*.log` per arm and per hypothesis, `metrics_*.txt`
summaries, `pdb_*/` output directories, `paired_cse.txt`.

## Next
Superseded by N025, which independently confirms P7/`SALAD_OPT_BATCHCOND` as
the WHOLE regression (0/160 diverged with it off, matching pre-regression
c5ab7a0 exactly) and localizes the mechanism this item left open (the exact
faulty kernel) to an inductor fusion miscompile fixable with
`max_fusion_size=1`.
