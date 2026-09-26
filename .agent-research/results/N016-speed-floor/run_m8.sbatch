#!/bin/bash -l
#SBATCH --job-name=n016_m8
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor/logs/m8_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor
cd /hnvme/workspace/v181bb10-salad/n016-floor
echo "### node $(hostname) $(date)"
timeout 2000 $P -u $R/x_triton_gemm.py 2>&1 | grep -E "^M8|Error|Traceback"
echo "########## D (weighted critical path)"
timeout 900 $P -u $R/x_census.py --preset exact --length 100 --sections D 2>&1 | grep -E "^CFG|^D "
echo "### done $(date)"
