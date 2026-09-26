#!/bin/bash -l
#SBATCH --job-name=n011_pre
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N011-fast-preset/logs/preset_%j.log
set -u
cd /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N011-fast-preset
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
run() { timeout 500 $P -u x_preset.py "$@" 2>&1 | grep -E "^RESULT|Error|Traceback" | tail -3; }
run --preset exact  --length 100
run --preset fast   --length 100
run --preset fast   --length 100 --override no_shared_draw
run --preset fast   --length 100 --override notf32
run --preset exact  --length 100 --override tf32
run --preset exact  --length 100
run --preset fast   --length 50
run --preset fast   --length 200
run --preset fast   --length 400
