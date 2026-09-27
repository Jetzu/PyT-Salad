#!/bin/bash -l
#SBATCH --job-name=salad_bc
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/outputs/benchc_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
L=50,100,200,400,800
echo "########## eager"
$P scripts/bench_torch.py --device cuda --lengths $L --iters 20 --out outputs/bc_eager.json
echo; echo "########## compile default"
$P scripts/bench_torch.py --device cuda --lengths $L --iters 20 --compile default --out outputs/bc_default.json
echo; echo "########## compile reduce-overhead"
$P scripts/bench_torch.py --device cuda --lengths $L --iters 20 --compile reduce-overhead --out outputs/bc_ro.json
