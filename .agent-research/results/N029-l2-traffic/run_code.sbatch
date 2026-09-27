#!/bin/bash -l
#SBATCH --job-name=n029_code
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N029-l2-traffic/logs/code_%j.log
set -u
module load cuda/12.6.2
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/ind_$SLURM_JOB_ID
export TRITON_CACHE_DIR=$TMPDIR/tri_$SLURM_JOB_ID
export TORCH_LOGS=output_code
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N029-l2-traffic
export SALAD_ROOT=/hnvme/workspace/v181bb10-salad/n029-l2
cd $SALAD_ROOT
echo "### node $(hostname) $(date)"
$P -u $R/scripts/x_code.py > $R/logs/outcode_$SLURM_JOB_ID.txt 2>&1
echo "--- exit $? size $(wc -l < $R/logs/outcode_$SLURM_JOB_ID.txt) lines ---"
gzip -f $R/logs/outcode_$SLURM_JOB_ID.txt
echo "### done $(date)"
rm -rf $TMPDIR/ind_$SLURM_JOB_ID $TMPDIR/tri_$SLURM_JOB_ID
