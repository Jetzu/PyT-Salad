#!/bin/bash -l
#SBATCH --job-name=n011_k32
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N011-fast-preset/logs/e2ek32_%j.log
set -u
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N011-fast-preset
echo "########## ARM fast_k32 (fast preset + K 64->32, 250/200)"
$P $R/x_gen_k32.py -- --num_aa 100 --num_designs 8 --fast --preset fast --seed 42 \
   --out_path $R/gen_fast_k32
$P $R/ss.py $R/gen_fast_k32
