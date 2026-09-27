#!/bin/bash -l
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --job-name=n029_gate
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N029-l2-traffic/logs/gate_%j.log
# PLACEMENT: cpu partition, 48 cores. The parity suite drives the EAGER path;
# GPU would only add nondeterminism (and K021 says these tests cannot see a
# compile-time change at all - that is what run_bits/run_quality are for).
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
cd /hnvme/workspace/v181bb10-salad/n029-l2
git rev-parse --short HEAD; md5sum salad_torch/geometry/features.py salad_torch/opt.py
run_gate () {
  echo "########## ARM $1"
  local fail=0
  for cfg in default_vp default_vp_scaled default_ve_scaled multimotif_vp; do
    echo "--- model_parity $cfg"
    SALAD_CONFIG=$cfg python -u tests/test_model_parity.py; e=$?; echo "exit=$e"
    [ $e -ne 0 ] && fail=1
  done
  for t in test_geometry_parity test_stack_parity test_encoder_parity test_sampling_parity; do
    echo "--- $t"; python -u tests/$t.py; e=$?; echo "exit=$e"
    [ $e -ne 0 ] && fail=1
  done
  echo "SUITE_EXIT=$fail ARM=$1"
}
SALAD_OPT_PAIRVEC=0 run_gate pairvec_off_BASELINE
SALAD_OPT_PAIRVEC=1 run_gate pairvec_on_O1_DEFAULT
