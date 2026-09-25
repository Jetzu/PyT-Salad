#!/bin/bash -l
#SBATCH --job-name=salad_gradstep
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:15:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N004-porting-gaps/gradstep_%j.log
set -ex
cd /home/hpc/v181bb/v181bb10/SALAD_torch

apptainer run --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
    .agent-research/results/N004-porting-gaps/gen_gradstep_ref.py \
    params/default_vp-200k.jax \
    .agent-research/results/N004-porting-gaps/gradstep_default_vp.npz
