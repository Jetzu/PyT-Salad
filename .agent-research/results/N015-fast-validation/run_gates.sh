#!/bin/bash
# N015 hard gates - the four checkpoint parity tests + geometry/stack/encoder/
# sampling. Takes the checkout to test as $1 so it can be pointed at the clean
# worktree (HEAD + only this task's change) instead of the shared main tree,
# which another agent is editing concurrently.
#
#   bash run_gates.sh /hnvme/workspace/v181bb10-salad/n015-worktree
#
# Floors (N005): 0.00002 A on default_vp / default_vp_scaled / default_ve_scaled,
# 0.00020 A on multimotif_vp; geometry 38/38.
set -u
R=${1:-/home/hpc/v181bb/v181bb10/SALAD_torch}
cd "$R" || exit 2
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
echo "gates on: $R   HEAD $(git rev-parse --short HEAD 2>/dev/null)"
rc=0
for C in default_vp default_vp_scaled default_ve_scaled multimotif_vp; do
  out=$(SALAD_CONFIG=$C $P tests/test_model_parity.py 2>&1); st=$?
  echo "$out" | grep -E "^coordinate deviation|^OK|^FAIL|FAILED|SKIP|aatype" | sed "s/^/[$C] /"
  echo "[$C] exit=$st"
  [ $st -ne 0 ] && rc=1
done
for T in test_geometry_parity test_stack_parity test_encoder_parity test_sampling_parity; do
  out=$($P tests/$T.py 2>&1); st=$?
  echo "$out" | tail -4 | sed "s/^/[$T] /"
  echo "[$T] exit=$st"
  [ $st -ne 0 ] && rc=1
done
echo "SUITE_EXIT=$rc"
exit $rc
