#!/bin/bash -l
#SBATCH --job-name=n016_cens
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor/logs/census_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor
cd /hnvme/workspace/v181bb10-salad/n016-floor
echo "### node $(hostname)  $(date)"
echo "########## P,F,B,D  exact, L=100  (one process)"
timeout 1500 $P -u $R/x_census.py --preset exact --length 100 --sections P,F,B,D 2>&1 | tail -90
echo "########## K  exact, L=100  (fresh process)"
timeout 1500 $P -u $R/x_census.py --preset exact --length 100 --sections K 2>&1 | tail -20
echo "########## K  fast,  L=100  (fresh process)"
timeout 1500 $P -u $R/x_census.py --preset fast  --length 100 --sections K 2>&1 | tail -20
echo "### done $(date)"
