#!/bin/bash -l
#SBATCH --job-name=n011_sp1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N011-fast-preset/logs/speed1_%j.log
set -u
cd /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N011-fast-preset
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader
for V in EXACT EXACT_AA BASE noaux topk cur48 cond32 both48 stride2 tf32 EXACT; do
  timeout 400 $P -u x_fast.py --variant $V --length 100 2>&1 | grep -E "^RESULT|Error|error|Traceback" | tail -3
done
