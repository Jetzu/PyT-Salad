#!/bin/bash -l
#SBATCH --job-name=n011_sp3
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N011-fast-preset/logs/speed3_%j.log
set -u
cd /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N011-fast-preset
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
for V in EXACT tf32 tf32med condsame tf32_condsame tf32_cur48 tf32_both32 tf32_condsame32 tf32 EXACT; do
  timeout 400 $P -u x_fast2.py --variant $V --length 100 2>&1 | grep -E "^RESULT|Error:|RuntimeError|Traceback" | tail -3
done
