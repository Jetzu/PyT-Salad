#!/bin/bash -l
#SBATCH --job-name=salad_n006_gcheck
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N006-codex-perf/graph_check_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u .agent-research/results/N006-codex-perf/verify_graph.py
