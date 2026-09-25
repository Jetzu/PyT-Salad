#!/bin/bash -l
#SBATCH --job-name=salad_flags
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N002-generation-flags/logs/verify_%j.log
set -u
cd /home/hpc/v181bb/v181bb10/SALAD_torch
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u \
  .agent-research/results/N002-generation-flags/verify_flags.py
