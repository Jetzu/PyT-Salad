#!/bin/bash -l
#SBATCH --job-name=salad_b1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/outputs/b1_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
for r in 1 2 3; do $P -u scripts/bench_one.py --length 100 --tag rep$r; done
for L in 50 200 400; do $P -u scripts/bench_one.py --length $L --tag len; done
