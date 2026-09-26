#!/bin/bash -l
#SBATCH --job-name=n016_r2d
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor/logs/r2d_%j.log
set -u
module load cuda/12.6.2
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_COMPILE_THREADS=1
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor
cd /hnvme/workspace/v181bb10-salad/n016-floor
echo "### node $(hostname) $(date)"
timeout 2000 ncu --nvtx --nvtx-include "measure/" \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum \
  --print-summary per-gpu --target-processes application-only \
  $P -u $R/x_ncu.py > $R/logs/ncu_raw.txt 2>&1
echo "--- exit $? ---"
grep -iE "dram__|lts__|Metric|kernels profiled|No kernels" $R/logs/ncu_raw.txt | head -20
echo "### done $(date)"
