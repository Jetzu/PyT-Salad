#!/bin/bash -l
#SBATCH --job-name=salad_gen
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/outputs/fullgen_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
mkdir -p outputs

echo "########## TORCH eager, L=100, 500 steps, 5 designs"
$P scripts/generate.py --num_aa 100 --num_designs 5 --out_path outputs/gen_torch \
   --timing_json outputs/gen_torch.json

echo; echo "########## TORCH compiled, L=100, 500 steps, 5 designs"
$P scripts/generate.py --num_aa 100 --num_designs 5 --compile default \
   --out_path outputs/gen_torch_c --timing_json outputs/gen_torch_c.json

echo; echo "########## JAX reference, same settings"
apptainer run --nv --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
  /salad/salad/training/eval_noise_schedule_benchmark.py \
    --config default_vp \
    --params /hnvme/workspace/v181bb10-salad/params/default_vp-200k.jax \
    --out_path outputs/gen_jax/ --num_aa 100 --num_designs 5 \
    --num_steps 500 --out_steps 400 --prev_threshold 0.8 \
    --timescale_pos "cosine(t)" 2>&1 | tail -20
