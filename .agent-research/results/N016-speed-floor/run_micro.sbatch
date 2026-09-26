#!/bin/bash -l
#SBATCH --job-name=n016_micro
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor/logs/micro_%j.log
set -u
module load cuda/12.6.2 2>/dev/null || true
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor
cd /hnvme/workspace/v181bb10-salad/n016-floor
echo "### node $(hostname)  $(date)"
echo "########## DEV + M1 (empty-graph replay)"
timeout 1200 $P -u $R/x_micro.py --only dev,m1 2>&1 | tail -80
echo "########## M2 (grid barrier)"
timeout 900 $P -u $R/x_micro.py --only m2 2>&1 | tail -40
echo "########## M3 (bandwidth)"
timeout 900 $P -u $R/x_micro.py --only m3 2>&1 | tail -30
echo "########## M4 (gemm)"
timeout 1200 $P -u $R/x_micro.py --only m4 2>&1 | tail -70
echo "### done $(date)"
