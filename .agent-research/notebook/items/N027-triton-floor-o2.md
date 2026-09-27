# N027 · Triton launch floor: O2 n=160 validation + the 331 single-op unfused kernels

type: task · date: 2026-09-27 · status: stub (no artifacts recoverable)

Backfilled by work-lint 2026-09-27: an INDEX row existed for N027 pointing at
a results folder that was never created, with no item file. See
`results/N027-triton-floor-o2/NOTE.md` for the full search (this repo,
`/hnvme/workspace/v181bb10-salad/`) and the cross-references from N028/N029/
N030 that confirm N027 was a real, separately-dispatched sibling task whose
output was simply never written down or committed under this ID (a
concurrent-agent ID-allocation race - N030's own item file records that its
brief was also originally labelled "N027" before it had to take N030 instead
because N027/N028/N029 were already claimed).

## Goal
(As titled in the orphaned INDEX row.) Presumably: finish N017/O2's `IPA
einsum -> mul+sum` validation at n=160, and explain/attack the 331 single-op
unfused Triton kernels N017 counted - the same general territory N028
explicitly deferred specific findings to ("N027's file", "N027's territory")
without claiming to have done that work itself.

## Plan
Unknown - not recoverable.

## Changes
None recoverable.

## Evidence
None recoverable.

## Outcome
No work product exists under this ID anywhere in the repo or on
`/hnvme/workspace/v181bb10-salad/`. This is a bookkeeping gap, not a negative
result - do not read "no evidence" as "O2/the 331-kernel census were
attempted and failed."

## Outputs
None.

## Next
If this scope still matters, re-run it as new work under a fresh ID. Do not
reuse N027 (its INDEX row/results path already point here as a closed stub).
