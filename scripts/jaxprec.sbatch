#!/bin/bash -l
#SBATCH --job-name=salad_jp
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:25:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/outputs/jaxprec_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
apptainer run --nv --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif scripts/jax_precision.py
