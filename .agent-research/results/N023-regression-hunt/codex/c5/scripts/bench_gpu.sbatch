#!/bin/bash -l
#SBATCH --job-name=salad_bench
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/outputs/bench_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
mkdir -p outputs
L=50,100,200,400,800
echo "########## PyTorch (GPU)"
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python scripts/bench_torch.py \
    --device cuda --lengths $L --iters 20 --out outputs/bench_torch_gpu.json
echo; echo "########## JAX (GPU)"
apptainer run --nv --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
    scripts/bench_jax.py --lengths $L --iters 20 --out outputs/bench_jax_gpu.json
