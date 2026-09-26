#!/bin/bash -l
#SBATCH --job-name=n021_roof
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching/logs/roof_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching
export SALAD_ROOT=/hnvme/workspace/v181bb10-salad/n021tree
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind-$SLURM_JOB_ID
L=${SALAD_L:?}
echo "### node $(hostname) $(date) L=$L"
for BG in ${SALAD_BG:?}; do
  B=${BG%%:*}; G=${BG##*:}
  timeout 1800 python -u $R/x_roofline.py --preset fast --length $L \
     --batch_size $B --group_size $G 2>&1 | grep -E "^ROOF|Error|Traceback|OutOfMemory"
done
echo "### done $(date)"
rm -rf $TORCHINDUCTOR_CACHE_DIR
