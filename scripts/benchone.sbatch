#!/bin/bash -l
#SBATCH --job-name=salad_b1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/outputs/b1_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
for L in 50 100 200 400 800; do $P -u scripts/bench_one.py --length $L; done
