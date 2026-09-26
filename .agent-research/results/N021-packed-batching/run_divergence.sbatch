#!/bin/bash -l
#SBATCH --job-name=n021_div
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching/logs/div_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching
export SALAD_ROOT=/hnvme/workspace/v181bb10-salad/n021tree
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind-$SLURM_JOB_ID
echo "### node $(hostname) $(date)"
# DETSEG=1: the only arm that can discriminate (N019 section 3).
for L in 100 200; do
  SALAD_OPT_DETSEG=1 timeout 2400 python -u $R/x_divergence.py --preset fast \
      --length $L --batch_size 4 2>&1 | grep -E "^#|^DIV|^FLOOR|Error|Traceback"
done
echo "### done $(date)"
rm -rf $TORCHINDUCTOR_CACHE_DIR
