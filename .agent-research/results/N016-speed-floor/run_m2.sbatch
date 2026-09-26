#!/bin/bash -l
#SBATCH --job-name=n016_m2
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:25:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor/logs/m2_%j.log
set -u
module load cuda/12.6.2
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export CUDA_HOME=${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}
echo "nvcc: $(which nvcc)  CUDA_HOME=$CUDA_HOME"
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N016-speed-floor
cd /hnvme/workspace/v181bb10-salad/n016-floor
timeout 900 $P -u $R/x_micro.py --only m2 2>&1 | tail -40
