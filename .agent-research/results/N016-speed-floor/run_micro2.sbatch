#!/bin/bash -l
#SBATCH --job-name=n016_m56
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:35:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor/logs/m56_%j.log
set -u
module load cuda/12.6.2
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor
cd /hnvme/workspace/v181bb10-salad/n016-floor
echo "### node $(hostname) $(date)  nvcc=$(which nvcc)"
timeout 1200 $P -u $R/x_micro2.py --only m5 2>&1 | grep -E "^M5|FAILED"
timeout 900  $P -u $R/x_micro2.py --only m6 2>&1 | grep -E "^M6|FAILED"
echo "### done $(date)"
