#!/bin/bash -l
#SBATCH --job-name=salad_distrib
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/outputs/distrib_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
echo "########## torch --fast, 40 designs"
$P scripts/generate.py --num_aa 100 --num_designs 40 --fast --seed 7 --out_path outputs/dist_torch
echo; echo "########## JAX reference, 40 designs"
apptainer run --nv --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
  /salad/salad/training/eval_noise_schedule_benchmark.py \
    --config default_vp --params /hnvme/workspace/v181bb10-salad/params/default_vp-200k.jax \
    --out_path outputs/dist_jax/ --num_aa 100 --num_designs 40 \
    --num_steps 500 --out_steps 400 --prev_threshold 0.8 --timescale_pos "cosine(t)" 2>&1 | tail -5
