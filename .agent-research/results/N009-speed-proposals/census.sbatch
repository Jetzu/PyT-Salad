#!/bin/bash -l
#SBATCH --job-name=salad_cen
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N009-speed-proposals/logs/cen_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
echo "=== kernel census, SHIPPING config (exact preset, all proposals on)"
$P -u .agent-research/results/N009-speed-proposals/x_kernels.py --length 100
echo "=== kernel census, every proposal OFF (the 4.95 ms starting point)"
env SALAD_OPT_GATHER=0 SALAD_OPT_HOIST=0 SALAD_OPT_NOPC=0 SALAD_OPT_LEAN=0 \
  $P -u .agent-research/results/N009-speed-proposals/x_kernels.py --length 100
echo "=== length sweep, shipping config"
for L in 50 200 400; do $P -u scripts/bench_one.py --length $L --no-eager --tag len$L; done
