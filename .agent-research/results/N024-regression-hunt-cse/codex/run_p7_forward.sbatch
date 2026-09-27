#!/bin/bash -l
#SBATCH --job-name=n024-p7fwd
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:05:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N024-regression-hunt-cse/codex/p7fwd_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_OPT_DETSEG=1
cd /home/hpc/v181bb/v181bb10/SALAD_torch
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u .agent-research/results/N024-regression-hunt-cse/codex/p7_forward_probe.py --device cuda --length 100 --chains 1
