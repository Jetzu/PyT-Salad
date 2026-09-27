#!/bin/bash -l
#SBATCH --job-name=n25p7u
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /hnvme/workspace/v181bb10-salad/n025_p7u_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
T=${TMPDIR:-/dev/shm}
export TORCHINDUCTOR_CACHE_DIR=$T/ind
export TRITON_CACHE_DIR=$T/triton
X=/hnvme/workspace/v181bb10-salad/n025_tools
H=/hnvme/workspace/v181bb10-salad/n025
W=$T/p7u; mkdir -p $W
echo "node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
echo "########## dump the same mid-rollout state (step 30 of 250)"
$P $X/x_p7.py --root $H --state $W/state.pt --dump 2>&1 | tail -3
echo "########## UNIT GPU fp32"
$P $X/x_p7u.py --root $H --state $W/state.pt 2>&1 | tail -12
echo "########## UNIT CPU (deterministic by construction)"
$P $X/x_p7u.py --root $H --state $W/state.pt --device cpu 2>&1 | tail -12
echo N025_P7U_DONE
