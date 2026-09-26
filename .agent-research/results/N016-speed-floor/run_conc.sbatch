#!/bin/bash -l
#SBATCH --job-name=n016_conc
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor/logs/conc_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor
cd /hnvme/workspace/v181bb10-salad/n016-floor
echo "### node $(hostname) $(date)"
for pre in exact fast; do
  echo "########## M9 $pre L=100"
  timeout 1300 $P -u $R/x_concurrency.py --preset $pre --length 100 --kmax 8 2>&1 | grep -E "^M9|Error|Traceback"
done
echo "### done $(date)"
