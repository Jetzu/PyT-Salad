#!/bin/bash
# N013 - the four checkpoint parity tests + geometry/stack/encoder/sampling.
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
rc=0
for C in default_vp default_vp_scaled default_ve_scaled multimotif_vp; do
  out=$(SALAD_CONFIG=$C $P tests/test_model_parity.py 2>&1)
  st=$?
  echo "$out" | grep -E "^coordinate deviation|^OK|^FAIL|FAILED|SKIP" | sed "s/^/[$C] /"
  echo "[$C] exit=$st"
  [ $st -ne 0 ] && rc=1
done
for T in test_geometry_parity test_stack_parity test_encoder_parity test_sampling_parity; do
  out=$($P tests/$T.py 2>&1); st=$?
  echo "$out" | tail -3 | sed "s/^/[$T] /"
  echo "[$T] exit=$st"
  [ $st -ne 0 ] && rc=1
done
echo "SUITE_EXIT=$rc"
