#!/bin/bash -l
#SBATCH --job-name=n029_graph
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N029-l2-traffic/logs/graph_%j.log
set -u
module load cuda/12.6.2
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/ind_$SLURM_JOB_ID
export TRITON_CACHE_DIR=$TMPDIR/tri_$SLURM_JOB_ID
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N029-l2-traffic
export SALAD_ROOT=/hnvme/workspace/v181bb10-salad/n029-l2
cd $SALAD_ROOT
echo "### node $(hostname) $(date)  HEAD $(git rev-parse --short HEAD)"
$P -u $R/scripts/x_graph.py
echo "--- graph exit $? ---"
echo "### baseline timing"
$P -u scripts/bench_one.py --length 100 --preset exact --no-eager --tag base
echo "### done $(date)"
rm -rf $TMPDIR/ind_$SLURM_JOB_ID $TMPDIR/tri_$SLURM_JOB_ID
