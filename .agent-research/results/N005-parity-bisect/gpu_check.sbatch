#!/bin/bash -l
#SBATCH --job-name=salad_n005
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:25:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N005-parity-bisect/gpu_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
R=.agent-research/results/N005-parity-bisect
P=/hnvme/workspace/v181bb10-salad/params
nvidia-smi -L
for c in default_vp default_vp_scaled default_ve_scaled; do
  apptainer run --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
      scripts/gen_fixtures_model.py $c $P/$c-200k.jax $R/gpu_$c.npz 48
  APPTAINERENV_JAX_DEFAULT_MATMUL_PRECISION=highest \
  apptainer run --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
      scripts/gen_fixtures_model.py $c $P/$c-200k.jax $R/gpuhi_$c.npz 48
done
apptainer run --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
    scripts/gen_fixtures_model.py multimotif_vp $P/multimotif_vp-200k.jax $R/gpu_multimotif_vp.npz 80
APPTAINERENV_JAX_DEFAULT_MATMUL_PRECISION=highest \
apptainer run --bind /hnvme:/hnvme --bind $PWD:$PWD salad.sif \
    scripts/gen_fixtures_model.py multimotif_vp $P/multimotif_vp-200k.jax $R/gpuhi_multimotif_vp.npz 80
echo DONE
