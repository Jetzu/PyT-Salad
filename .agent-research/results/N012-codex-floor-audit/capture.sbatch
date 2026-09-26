#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N012-codex-floor-audit/capture_%j.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export TORCHINDUCTOR_CACHE_DIR=/tmp/salad_n012_capture_${SLURM_JOB_ID}
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u .agent-research/results/N012-codex-floor-audit/capture_compare.py
