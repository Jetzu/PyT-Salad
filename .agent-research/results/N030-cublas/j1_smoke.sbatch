#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=n030_smoke
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas/j1_smoke_%j.log
# PLACEMENT: 1x H100, 16 cores, <=2 GPU-h. GPU-only: the whole point is which
# CUDA kernel inductor selects, which does not exist on CPU.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n030
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCHINDUCTOR_PRINT_AUTOTUNE=1
export PYTHONHASHSEED=0
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas
cd $SALAD_TREE
git -C $SALAD_TREE rev-parse HEAD; md5sum salad_torch/gemm_exact.py
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
for ARM in exact all; do
  echo "########## SMOKE ARM=$ARM"
  N030_ARM=$ARM TORCHINDUCTOR_CACHE_DIR=/dev/shm/n030_${SLURM_JOB_ID}_s_$ARM \
    python -u $R/x_smoke.py || true
  rm -rf /dev/shm/n030_${SLURM_JOB_ID}_s_$ARM
done
echo "########## BITS L=64"
SALAD_OPT_DETSEG=1 N030_LEN=64 N030_ARMS=base,all,exact,base_rerun \
  TORCHINDUCTOR_CACHE_DIR=/dev/shm/n030_${SLURM_JOB_ID}_b64 \
  python -u $R/x_bits.py || true
rm -rf /dev/shm/n030_${SLURM_JOB_ID}_*
