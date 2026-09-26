#!/bin/bash -l
#SBATCH --job-name=n016_r7
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor/logs/r7_%j.log
set -u
module load cuda/12.6.2 2>/dev/null || true
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor
cd /hnvme/workspace/v181bb10-salad/n016-floor
echo "### node $(hostname) $(date)"
echo "########## R7: non-matmul arithmetic on the lean step"
timeout 900 $P -u $R/x_census.py --preset exact --length 100 --sections D 2>&1 | grep -E "^D (NON|weighted|graph|aten|flop)"
echo "########## R2: does ncu work here?"
which ncu && timeout 600 ncu --version 2>&1 | head -3 || echo "NCU-UNAVAILABLE"
echo "### done $(date)"
