#!/bin/bash -l
#SBATCH --job-name=salad_n006_cmp
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:35:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N006-codex-perf/compare_modes_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u .agent-research/results/N006-codex-perf/compare_modes.py
