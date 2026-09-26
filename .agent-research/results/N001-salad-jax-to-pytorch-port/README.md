# N001 results

N001 is the umbrella port task; see `notebook/items/N001-salad-jax-to-pytorch-port.md`
for the goal/plan/evidence. It has no dedicated experiment outputs of its own -
its artifacts are the repo's permanent code and docs (`salad_torch/`, `notes/`,
`tests/`, `fixtures/*.npz`, `outputs/gen_{torch,jax}/`) plus the child items
that carry it forward: N002 (generation flags), N003 (red-team), N004
(porting gaps), N005 (parity bisect), N006/N009/N011/N013/N014/N015 (speed),
N007/N010/N012 (speed plan / red-team on the fast path).

Backfilled by work-lint 2026-09-26: this folder did not exist even though the
INDEX row cited it.
