#!/bin/bash -l
#SBATCH --job-name=n011_e2e
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:55:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N011-fast-preset/logs/e2e_%j.log
set -u
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N011-fast-preset
N=8
echo "########## ARM exact (500/400)"
$P scripts/generate.py --num_aa 100 --num_designs $N --fast --preset exact --seed 42 \
   --out_path $R/gen_exact
echo "########## ARM fast_full (fast numerics, 500/400)"
$P scripts/generate.py --num_aa 100 --num_designs $N --fast --preset fast --seed 42 \
   --num_steps 500 --out_steps 400 --out_path $R/gen_fast_full
echo "########## ARM fast (250/200)"
$P scripts/generate.py --num_aa 100 --num_designs $N --fast --preset fast --seed 42 \
   --out_path $R/gen_fast
echo "########## ARM fast_125 (125/100)"
$P scripts/generate.py --num_aa 100 --num_designs $N --fast --preset fast --seed 42 \
   --num_steps 125 --out_steps 100 --out_path $R/gen_fast125
echo "########## QUALITY GATE"
$P $R/ss.py $R/gen_exact $R/gen_fast_full $R/gen_fast $R/gen_fast125
