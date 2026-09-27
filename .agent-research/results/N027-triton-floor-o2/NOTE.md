# N027 — no artifacts found (work-lint 2026-09-27)

The INDEX row for N027 ("Triton launch floor: O2 n=160 validation + the 331
single-op unfused kernels") pointed at a results folder
(`N027-triton-launch-floor-o2-n-160-validation/`) that does not exist, and no
item file existed. This lint searched `.agent-research/`, the whole git
working tree (tracked + untracked), and `/hnvme/workspace/v181bb10-salad/` for
any job logs, scripts, or a worktree under this ID and found nothing.

N027's scope is real and independently attested: `results/N028-triton-floor/
RESULTS.md` twice defers specific findings to "N027's [territory/file]"
(the HkLinear-bias and LayerNorm single-op kernel counts, and per-block
weight-GEMM batching), and `results/N029-l2-traffic/RESULTS.md` lists the
`einsum -> mul+sum` (O2) lever as owned by "N027 (O2)" among un-landed
opportunities. `results/N030-cublas/RESULTS.md` and
`notebook/items/N030-cublas.md` independently confirm the mechanism: N030's
brief was originally also labelled "N027", but by the time that agent went to
create its notebook entry, IDs N027/N028/N029 had already been claimed by
concurrent sibling agents, so it took N030 instead. This is a multi-agent ID
race: whichever agent was actually dispatched as "N027" appears to have
reserved the INDEX row (or had it reserved for it) but never produced or
committed any output under that ID - it is not recoverable from what is on
disk today.

No results are fabricated here. If the N027-scoped work (O2 n=160 validation,
331-kernel census) still matters, it should be re-run as new work under a
fresh ID rather than assumed complete.
