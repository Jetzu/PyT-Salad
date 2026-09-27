#!/bin/bash -l
#SBATCH --job-name=n029_ncu2
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N029-l2-traffic/logs/ncu_%j.log
set -u
module load cuda/12.6.2
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/ind_$SLURM_JOB_ID
export TRITON_CACHE_DIR=$TMPDIR/tri_$SLURM_JOB_ID
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N029-l2-traffic
export SALAD_ROOT=/hnvme/workspace/v181bb10-salad/n029-l2
cd $SALAD_ROOT
echo "### node $(hostname) $(date)  HEAD $(git rev-parse --short HEAD)"
timeout 3000 ncu --nvtx --nvtx-include "measure/" \
  --metrics lts__t_bytes.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum \
  --print-summary per-gpu --target-processes application-only \
  $P -u $R/scripts/x_ncu.py > $R/logs/ncu_raw_$SLURM_JOB_ID.txt 2>&1
echo "--- ncu exit $? ---"
tail -5 $R/logs/ncu_raw_$SLURM_JOB_ID.txt
echo "### done $(date)"
rm -rf $TMPDIR/ind_$SLURM_JOB_ID $TMPDIR/tri_$SLURM_JOB_ID
