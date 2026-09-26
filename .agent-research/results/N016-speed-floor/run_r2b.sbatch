#!/bin/bash -l
#SBATCH --job-name=n016_r2b
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor/logs/r2c_%j.log
set -u
module load cuda/12.6.2
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_COMPILE_THREADS=1
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor
cd /hnvme/workspace/v181bb10-salad/n016-floor
echo "### node $(hostname) $(date)"
timeout 2400 ncu --nvtx --nvtx-include "measure/" --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum \
  --print-summary per-gpu --target-processes application-only $P -u $R/x_ncu.py 2>&1 \
  | grep -viE "^E[0-9]{4}|Traceback|File \"|    " | tail -30
echo "### done $(date)"
