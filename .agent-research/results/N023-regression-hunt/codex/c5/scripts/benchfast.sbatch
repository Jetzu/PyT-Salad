#!/bin/bash -l
#SBATCH --job-name=salad_bf
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/outputs/bf_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python scripts/bench_fast.py --lengths 50,100,200,400
