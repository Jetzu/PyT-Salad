#!/bin/bash -l
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --job-name=n030_gate
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas/j4_gate_%j.log
# PLACEMENT: cpu partition, 48 cores (full NUMA fraction). The parity suite is
# the EAGER path - GPU would only add nondeterminism and buy nothing. Note the
# structural caveat this project already knows: the suite CANNOT see a
# compile-time kernel-selection change (K021), so it is necessary, not
# sufficient; j5 (design quality) is the gate that can.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
cd /hnvme/workspace/v181bb10-salad/n030
md5sum salad_torch/modules/basic.py salad_torch/opt.py salad_torch/gemm_exact.py salad_torch/modules/fast.py
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
run_gate ship_default
SALAD_OPT_GEMMTPL=exact run_gate explicit_exact
