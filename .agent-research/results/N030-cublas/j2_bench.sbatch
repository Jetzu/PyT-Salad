#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=n030_bench
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas/j2_bench_%j.log
# PLACEMENT: 1x H100, 16 cores, <=4 GPU-h. GPU-only - the whole question is
# which CUDA kernel inductor picks, which has no CPU analogue. ONE arm and ONE
# length per python, per the N009/N012 method note.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n030
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCHINDUCTOR_PRINT_AUTOTUNE=1
export PYTHONHASHSEED=0
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas
cd $SALAD_TREE
git -C $SALAD_TREE rev-parse HEAD; md5sum salad_torch/gemm_exact.py salad_torch/modules/basic.py
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
L=${N030_LEN:-100}
for ARM in off all exact decomp32 exact+decomp32; do
  echo "########## ARM=$ARM L=$L"
  N030_TAG=${ARM}_L${L} N030_LEN=$L N030_ARM=$ARM N030_REPS=3 \
    SALAD_OPT_GEMMTPL=1 SALAD_GEMM_EXACT_LOG=1 \
    TORCHINDUCTOR_CACHE_DIR=/dev/shm/n030_${SLURM_JOB_ID}_${ARM}_${L} \
    python -u $R/x_bench.py || true
  rm -rf /dev/shm/n030_${SLURM_JOB_ID}_${ARM}_${L}
done
rm -rf /dev/shm/n030_${SLURM_JOB_ID}_*
