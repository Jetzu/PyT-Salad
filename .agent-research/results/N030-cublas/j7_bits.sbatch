#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --job-name=n030_bits2
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas/j7_bits_%j.log
# PLACEMENT: 1x H100, 16 cores, <=6 GPU-h. GPU-only by construction: Triton
# matmul templates do not exist on the CPU backend (N020 3a), so the CPU
# bitwise control is a trivial pass and says nothing. Under SALAD_OPT_DETSEG=1
# the only CUDA run-to-run nondeterminism in the model is gone, which is what
# makes a GPU bitwise comparison meaningful.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n030
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export PYTHONHASHSEED=0
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas
cd $SALAD_TREE
git -C $SALAD_TREE rev-parse HEAD; md5sum salad_torch/gemm_exact.py
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
for L in 100; do
  echo "########## BITS L=$L"
  SALAD_OPT_DETSEG=1 N030_LEN=$L \
    N030_ARMS=base,exact+noepi,exact+noepi+decomp32,decomp32,base_rerun \
    SALAD_GEMM_EXACT_LOG=1 \
    TORCHINDUCTOR_CACHE_DIR=/dev/shm/n030_${SLURM_JOB_ID}_b${L} \
    python -u $R/x_bits.py || true
  rm -rf /dev/shm/n030_${SLURM_JOB_ID}_b${L}
done
rm -rf /dev/shm/n030_${SLURM_JOB_ID}_*
