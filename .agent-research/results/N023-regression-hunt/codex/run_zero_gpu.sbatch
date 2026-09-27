#!/bin/bash -l
#SBATCH --job-name=n023_zerogpu
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:10:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex/zerogpu_%j.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
S=.agent-research/results/N023-regression-hunt/codex/x_p7_zero.py
for detseg in 0 1; do
    export SALAD_OPT_DETSEG=$detseg
    echo "### DETSEG=$detseg zero prev, L=100"
    $P "$S" --device cuda --length 100
    echo "### EXIT=$?"
done
