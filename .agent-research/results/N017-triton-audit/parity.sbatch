#!/bin/bash -l
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --job-name=n017_parity
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N017-triton-audit/parity_%j.log
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
cd /hnvme/workspace/v181bb10-salad/n017live
md5sum salad_torch/geometry/segment.py salad_torch/modules/attention.py
# ARM = the env the tests run under; each arm runs the full gate.
run_gate () {
  echo "########## ARM $1"
  for cfg in default_vp default_vp_scaled default_ve_scaled multimotif_vp; do
    echo "--- model_parity $cfg"; SALAD_CONFIG=$cfg python -u tests/test_model_parity.py; echo "exit=$?"
  done
  for t in test_geometry_parity test_stack_parity test_encoder_parity test_sampling_parity; do
    echo "--- $t"; python -u tests/$t.py; echo "exit=$?"
  done
}
N017_IDXMEAN=0 N017_IPA=0 run_gate baseline
N017_IDXMEAN=1 N017_IPA=0 run_gate idxmean
N017_IDXMEAN=1 N017_IPA=1 run_gate idxmean+ipa
