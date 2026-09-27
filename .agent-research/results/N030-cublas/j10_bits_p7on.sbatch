#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=n030_bitsp7
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas/j10_bits_p7on_%j.log
# PLACEMENT: 1x H100, 16 cores, <2 GPU-h. Whole-forward bitwise check of
# decompose_small_bmm on top of the CURRENT shipped baseline (P7 re-enabled,
# fusion barrier). Same methodology as j7_bits (ONE process, distinct
# joint_custom_post_pass names per arm to salt the FX cache, DETSEG=1).
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n030
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export PYTHONHASHSEED=0
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas
cd $SALAD_TREE
git -C /home/hpc/v181bb/v181bb10/SALAD_torch rev-parse HEAD
md5sum salad_torch/gemm_exact.py salad_torch/modules/fast.py salad_torch/opt.py salad_torch/modules/cond_prologue.py
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
for L in 100; do
  echo "########## BITS L=$L (P7 ON)"
  SALAD_OPT_DETSEG=1 N030_LEN=$L \
    N030_ARMS=base,decomp32,base_rerun \
    SALAD_GEMM_EXACT_LOG=1 \
    TORCHINDUCTOR_CACHE_DIR=/dev/shm/n030_${SLURM_JOB_ID}_b${L} \
    python -u $R/x_bits.py || true
  rm -rf /dev/shm/n030_${SLURM_JOB_ID}_b${L}
done
