#!/bin/bash -l
#SBATCH --job-name=n016_c400
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor/logs/conc400_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor
cd /hnvme/workspace/v181bb10-salad/n016-floor
echo "### node $(hostname) $(date)"
echo "########## M9 exact L=400 (does the concurrency headroom survive a bigger protein?)"
timeout 1400 $P -u $R/x_concurrency.py --preset exact --length 400 --kmax 4 2>&1 | grep -E "^M9|Error|Traceback"
echo "########## D at L=400 (is the critical-path depth length-independent?)"
timeout 1200 $P -u $R/x_census.py --preset exact --length 400 --sections D 2>&1 | grep -E "^CFG|^D (aten|weighted|graph|level)"
echo "### done $(date)"
