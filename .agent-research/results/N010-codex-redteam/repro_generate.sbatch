#!/bin/bash
#SBATCH --job-name=n010-fast-e2e
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N010-codex-redteam/repro_generate_slurm.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export PYTHONPATH=.
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u scripts/generate.py --fast --device cuda --num_aa 48 --num_designs 1 --num_steps 2 --out_steps 1 --out_path .agent-research/results/N010-codex-redteam/generated
