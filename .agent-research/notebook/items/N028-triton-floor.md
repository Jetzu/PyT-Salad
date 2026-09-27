# N028 · Triton launch floor: O2 at n=160, the single-op kernels, and the 3x3 bmm

type: task · date: 2026-09-27 · status: in-progress (O6 landed; O8/O2 pending)

## Goal
Recover the ~250-350 us of removable Triton launch floor N017 identified but
found no safe route to (8-10 % of the 3.375 ms lean step at L=100).
Three deliverables: (1) finish N017/O2 with an n=160 structural verdict,
(2) explain the 331 single-op unfused Triton kernels and fix what is fixable
from the Python side, (3) anything else, costed honestly.

## Plan
Worktree `/hnvme/workspace/v181bb10-salad/n028wt`, detached at 98bc61b.
Four flags, each its own env var, read at import time, one process per arm.

* **O2** `SALAD_OPT_IPAMULSUM` - N017's IPA einsum -> mul+sum. Reassociation.
* **O6** `SALAD_OPT_PAIRFEAT` - `direction_features` is a SLICE of
  `pair_vector_features`' intermediate; compute both in one pass. Bit-exact.
* **O8** `SALAD_OPT_FRAMEMULSUM` - `apply_t`/`apply_inverse_t`/`rot_compose`
  are `einsum`, and ATen lowers every one to `bmm`: 61 of 303 `extern_kernels.bmm`
  calls have a `(...,3,3)` operand. Reassociation over a 3-term contraction.
* fusion-refusal census: `TORCH_LOGS=fusion` + `ir_pre/post_fusion` + the
  profiler, to rank WHY 331 kernels carry one op.

## Changes
Landing picked up on branch `n006-manual-cudagraph` against current HEAD
(40a85be, P7 re-enabled - the measurements above predate that and needed
re-baselining). **O6 as originally planned (share the WIDE `pair_vector_features`
contraction, slice `direction_features` out of it) is WRONG** - see RESULTS.md
section 8. Landed instead: a narrower `direction_and_pair_vector_features`
(`salad_torch/geometry/features.py`) that shares only `aa_frames_t(pos)` and
the neighbour gather (both rounding-free) and calls each function's own
`apply_inverse_t`/`vec_normalized` unchanged in width and call count.
`opt.PAIR_FEAT_MERGE` is a proper preset key now (`True` in every preset,
veto-only via `SALAD_OPT_PAIRFEAT=0`, matching `BATCH_COND`'s contract) rather
than a bare module default. Also fixed in the same pass: `IPA_MULSUM`/
`FRAME_MULSUM` (O2/O8, staged but not yet landed) had the veto polarity
backwards (`preset OR env`, a measurement-phase leftover) - flipped to
`preset AND env`, matching the project convention, before anything shipped.

## Evidence
* Bug found: two-process, block-by-block hook on `DiffusionPairFeatures.forward`
  - block 0/1 exact, **block 2 diverges 1.8e-7 with bit-identical input `pos`**;
  a fresh process replaying the exact captured tensors reproduces the reference
  exactly - call-context-dependent BLAS kernel selection, not a value/logic bug.
  Same mechanism N029 found independently in its "O1" (its RESULTS.md §5.2-5.3).
* Fix verified bit-exact: compiled `forward_core` CPU control 0.0/0.0 at L=64
  and L=100 (job 907814); eager `test_sampling_parity` produces
  BIT-IDENTICAL trajectories at `SALAD_OPT_PAIRFEAT=0` vs `=1` (1/64, 7.3x,
  matching digit-for-digit - not just both passing) - jobs 907814 and 907816.
* Full gate, current HEAD, corrected code: parity suite (4 configs + geometry/
  stack/encoder/sampling) EXIT 0 on CPU (907814, 907816) and GPU (907815);
  `test_design_quality.py --n 160 --preset fast` 0/160 diverged; `--n 64
  --preset exact` 0/64 diverged (907815).
* O8's CPU bitwise control re-confirmed NOT bit-exact on the corrected O6
  baseline (9/11 differ, job 907816) - matches section 4's original finding,
  unaffected by the O6 fix.

## Outcome
**O6 landed** (commit follows this notebook update) - bit-exact, ships in
every preset, but the honest measured win is close to zero (3.076-3.127 ms
vs the pre-O6 3.074 ms HEAD baseline, well inside 2-rep noise) because the
mechanism that produced the original -5.0% claim (removing the wide
`apply_inverse_t` bmm) is exactly what N029 proved is not bit-exact. It ships
anyway: free, harmless, and correct.
**O8 and O2 not yet landed** - code is staged (frames.py/rigid.py for O8,
attention.py for O2), gated off by default (`frame_mulsum`/`ipa_mulsum` False
in every preset pending their own n=160 confirmation under the corrected
veto-polarity code, which differs mechanically from how they were measured
in sections 1 and 4 above even though the arithmetic is identical). Next.

## Outputs
Raw SLURM logs for this item's jobs are in `results/N028-triton-floor/logs/`,
except `fus_907103.log` (96 MiB, >5MB cap) which lint moved to
`/hnvme/workspace/v181bb10-salad/n028_evidence/fus_907103.log`.

## Next
