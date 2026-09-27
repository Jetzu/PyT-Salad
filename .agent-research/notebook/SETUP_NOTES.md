# Setup notes / ambiguity decisions

- **2026-09-27 (N035)**: `new-item.sh` flagged `LINT DUE` (19 items since last lint) when
  creating the N035 item. Deliberately did NOT run `work-lint` at that point: two sibling
  agents (shape-conditioning `--blobs` / N034, a symmetry optimiser) had live, uncommitted
  edits in the shared notebook at the time (`INDEX.md` itself, plus their own in-progress
  item files) alongside uncommitted code changes in the same working tree. A lint pass
  consolidates/rewrites across the WHOLE notebook, which risks clobbering their in-flight
  work or racing their own eventual commits. Conservative choice per this file's own
  purpose: defer, don't guess. Whoever lands next after all three sibling tasks (N034, N035,
  and the symmetry optimiser's - number not yet known to this agent) have committed their
  own work should run `work-lint`.
