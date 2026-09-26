#!/bin/bash -l
#SBATCH --job-name=n021_parity
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching/logs/parity_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
cd /hnvme/workspace/v181bb10-salad/n021tree
echo "### node $(hostname) $(date)"
md5sum salad_torch/modules/fast.py salad_torch/diffusion/sampling.py scripts/generate.py
rc=0
for C in default_vp default_vp_scaled default_ve_scaled multimotif_vp; do
  out=$(SALAD_CONFIG=$C python tests/test_model_parity.py 2>&1); st=$?
  echo "$out" | grep -E "^coordinate deviation|^OK|^FAIL|FAILED|SKIP" | sed "s/^/[$C] /"
  echo "[$C] exit=$st"; [ $st -ne 0 ] && rc=1
done
for T in test_geometry_parity test_stack_parity test_encoder_parity test_sampling_parity; do
  out=$(python tests/$T.py 2>&1); st=$?
  echo "$out" | tail -3 | sed "s/^/[$T] /"
  echo "[$T] exit=$st"; [ $st -ne 0 ] && rc=1
done
echo "SUITE_EXIT=$rc"
echo "### done $(date)"
