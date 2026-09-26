#!/bin/bash
#SBATCH --job-name=n010-fast-recheck
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N010-codex-redteam/repro_fast_recheck.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export PYTHONPATH=.
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u .agent-research/results/N010-codex-redteam/repro_fast_gpu.py
