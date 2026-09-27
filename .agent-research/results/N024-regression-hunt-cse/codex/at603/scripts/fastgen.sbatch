#!/bin/bash -l
#SBATCH --job-name=salad_fastgen
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/outputs/fastgen_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
echo "########## baseline (eager) 3 designs"
$P scripts/generate.py --num_aa 100 --num_designs 3 --out_path outputs/gen_base
echo; echo "########## --fast, 5 designs"
$P scripts/generate.py --num_aa 100 --num_designs 5 --fast --out_path outputs/gen_fast
