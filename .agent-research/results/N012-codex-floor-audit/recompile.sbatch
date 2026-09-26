#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N012-codex-floor-audit/recompile_%j.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export TORCHINDUCTOR_CACHE_DIR=/tmp/salad_n012_recompile_${SLURM_JOB_ID}
export TORCH_LOGS=recompiles
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u .agent-research/results/N012-codex-floor-audit/recompile_audit.py
