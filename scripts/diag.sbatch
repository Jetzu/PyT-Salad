#!/bin/bash -l
#SBATCH --job-name=salad_diag
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/outputs/diag_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u scripts/prof_kernels.py
