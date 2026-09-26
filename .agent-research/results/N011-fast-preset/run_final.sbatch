#!/bin/bash -l
#SBATCH --job-name=n011_fin
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:55:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N011-fast-preset/logs/final_%j.log
set -u
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N011-fast-preset
echo "########## SPEED on the final code"
timeout 700 $P -u $R/x_preset.py --preset exact --length 100 2>&1 | grep -E "^RESULT|Error" | tail -2
timeout 700 $P -u $R/x_preset.py --preset fast  --length 100 2>&1 | grep -E "^RESULT|Error" | tail -2
echo "########## QUALITY on the final preset (8 designs)"
$P scripts/generate.py --num_aa 100 --num_designs 8 --fast --preset fast --seed 42 \
   --out_path $R/gen_fast_final
$P $R/ss.py $R/gen_fast_final $R/gen_exact
