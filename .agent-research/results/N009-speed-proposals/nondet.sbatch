#!/bin/bash -l
#SBATCH --job-name=salad_nd
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:25:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N009-speed-proposals/logs/nd_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
echo "=== is the CUDA denoiser reproducible at all?"
$P -u .agent-research/results/N009-speed-proposals/x_nondet.py --device cuda
echo "=== equivalence harness, GPU, with its own noise-floor control"
$P -u .agent-research/results/N009-speed-proposals/x_equiv.py --device cuda
