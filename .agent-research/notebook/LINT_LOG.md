# Lint log

## 2026-09-26 (first lint, last_lint_item N000 -> N015)

Counts: 19 check.sh problems fixed (8 files >5MB moved to
`large_output_location` + 2 citations repointed to the new absolute paths;
1 missing results folder backfilled [N001]; 3 missing item files backfilled
[N007, N010, N012] which then surfaced 3 more missing-INDEX-row problems,
fixed by adding those rows; 4 more >5MB files appeared mid-lint, in two
waves, from concurrently-running N013 SLURM jobs [903260/903261, then
903260 continuing] and were moved the same way - N013's folder was an
actively moving target throughout this lint, jobs 903260/903261/903262/
903291/903293 all still `R` in squeue when this lint finished; re-run
check.sh before trusting N013 clean on any later lint) - 1 merged (K005 absorbed N010/S7 as a second independent confirmation
of the same clash-clamp bug, cites 1->2) - 0 superseded (R001 was already
superseded pre-lint; no new contradictions found) - 0 promoted from
[unverified] (none were tagged unverified) - 4 marked [stale, fixed] (K001,
K003, K004, K006 - all read against current source and found already fixed:
encoder.py's Rigid indexing, fast.py's cache key and static-buffer clone,
model.py's cloud_std override) - 1 rule promoted to RULES.md (TF32 GPU-JAX-
fixture trap, cited by 6 items: N001/N005/N007/N011/N012/N015).

Also: rewrote KNOWLEDGE.md's Headline (10 bullets); backfilled N001's
results folder and N007/N010/N012's item files + INDEX rows (all three were
Codex/plan outputs that had sat with zero notebook entry); added TODO T9-T14
for N010/N012 findings not yet triaged (S3 symmetry/PDB-writer bug, S4/S5/T11
verification-suite gaps, S6/C1/C2 minor crashes, N012's floor-claim
correction into N009/N014) and updated T1-T3/T4/T5/T6's status against a
direct read of current source (T1-T3 done, T4/T6 partially done, T5 has a
second independent confirmation).

Left unresolved: T9-T13 (read-only observations, not first-hand-reproduced -
per policy, no code change without reproducing first); N013's item file is
still a stub (work is actively in-progress under running SLURM jobs
903260-903262 as of this lint - not the lint's job to fill in); N012's
`latest/` is a full nested clone of the repo at commit e2a3464 (including
its own `.agent-research/` snapshot) - harmless per check.sh (no file inside
it exceeds 5MB) but worth a human glance if disk use ever matters.
