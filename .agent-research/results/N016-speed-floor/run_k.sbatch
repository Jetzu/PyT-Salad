#!/bin/bash -l
#SBATCH --job-name=n016_k
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor/logs/k_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor
cd /hnvme/workspace/v181bb10-salad/n016-floor
echo "### node $(hostname) $(date)"
for pre in exact fast; do
  echo "########## K $pre"
  timeout 1200 $P -u $R/x_census.py --preset $pre --length 100 --sections K 2>&1 | grep -E "^CFG|^K"
done
echo "### done $(date)"
