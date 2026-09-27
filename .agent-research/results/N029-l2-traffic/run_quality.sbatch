#!/bin/bash -l
#SBATCH --job-name=n029_qual
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N029-l2-traffic/logs/qual_%j.log
# PLACEMENT: 1 h100. This gate GENERATES 160 designs through the compiled+captured
# path, which only exists on GPU; ~20 min + one compile.
set -u
module load cuda/12.6.2
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/ind_$SLURM_JOB_ID
export TRITON_CACHE_DIR=$TMPDIR/tri_$SLURM_JOB_ID
cd /hnvme/workspace/v181bb10-salad/n029-l2
git rev-parse --short HEAD; md5sum salad_torch/geometry/features.py
echo "### node $(hostname) $(date)"
python -u tests/test_design_quality.py --n 160 --preset fast; echo "QUALITY_EXIT=$?"
echo "### done $(date)"
rm -rf $TMPDIR/ind_$SLURM_JOB_ID $TMPDIR/tri_$SLURM_JOB_ID
