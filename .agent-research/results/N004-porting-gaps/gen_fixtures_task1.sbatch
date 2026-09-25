#!/bin/bash -l
#SBATCH --job-name=salad_fix1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:20:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N004-porting-gaps/genfix_%j.log
set -ex
cd /home/hpc/v181bb/v181bb10/SALAD_torch

apptainer run --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
    scripts/gen_fixtures_model.py default_vp_scaled \
    /hnvme/workspace/v181bb10-salad/params/default_vp_scaled-200k.jax \
    fixtures/model_default_vp_scaled.npz 48

apptainer run --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
    scripts/gen_fixtures_model.py multimotif_vp \
    /hnvme/workspace/v181bb10-salad/params/multimotif_vp-200k.jax \
    fixtures/model_multimotif_vp.npz 80
