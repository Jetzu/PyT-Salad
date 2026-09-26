#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N012-codex-floor-audit/latest_graph_%j.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N012-codex-floor-audit/latest
export TORCHINDUCTOR_CACHE_DIR=/tmp/salad_n012_latest_graph_${SLURM_JOB_ID}
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u ../graph_audit.py --config "$1"
