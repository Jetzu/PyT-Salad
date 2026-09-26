#!/bin/bash -l
#SBATCH --job-name=n021_ncu
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching/logs/ncu_%j.log
set -u
module load cuda/12.6.2
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching
export SALAD_ROOT=/hnvme/workspace/v181bb10-salad/n021tree
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind-$SLURM_JOB_ID
cd $SALAD_ROOT
echo "### node $(hostname) $(date)"
for B in 1 8; do
  echo "########## batch_size=$B, L=100, one lean forward (uncaptured, NVTX-scoped)"
  SALAD_B=$B SALAD_L=100 timeout 2400 ncu --nvtx --nvtx-include "measure/" \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum \
    --print-summary per-gpu --target-processes all $P -u $R/x_ncu_packed.py 2>&1 | tail -25
done
echo "### done $(date)"
rm -rf $TORCHINDUCTOR_CACHE_DIR
