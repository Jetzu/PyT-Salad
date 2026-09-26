#!/bin/bash -l
#SBATCH --job-name=n016_r2
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor/logs/r2_%j.log
set -u
module load cuda/12.6.2
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor
cd /hnvme/workspace/v181bb10-salad/n016-floor
echo "### node $(hostname) $(date)"
echo "########## R2a: minimum materialised traffic from the graph"
timeout 900 $P -u $R/x_census.py --preset exact --length 100 --sections D 2>&1 | grep -E "^D (MIN|NON|weighted|graph|aten)"
echo "########## R2b: measured DRAM traffic (ncu, nvtx-scoped to one lean forward)"
timeout 1800 ncu --nvtx --nvtx-include "measure/" --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum \
  --print-summary per-gpu --target-processes all $P -u $R/x_ncu.py 2>&1 | tail -40
echo "### done $(date)"
