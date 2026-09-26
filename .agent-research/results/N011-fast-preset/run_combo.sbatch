#!/bin/bash -l
#SBATCH --job-name=n011_cmb
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:55:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N011-fast-preset/logs/combo_%j.log
set -u
cd /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N011-fast-preset
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
for V in base topk combo combo_topk; do
  echo "---------- $V"
  timeout 800 $P -u x_combo.py --variant $V --length 100 2>&1 \
    | grep -E "^RESULT|Error|Exception|Traceback|NotImplemented|assert" | tail -4
done
