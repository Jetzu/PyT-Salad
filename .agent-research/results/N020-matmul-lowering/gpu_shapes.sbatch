#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=n020_shapes
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering/gpu_shapes_%j.log
# PLACEMENT: 1x H100, 16 cores. GPU-ONLY by nature - the whole question is which
# CUDA kernel is faster at each of the model's real GEMM shapes, measured inside
# a captured graph. ~2 GPU-h budgeted (est. well under 25).
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n020
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering
cd $SALAD_TREE
md5sum salad_torch/modules/basic.py salad_torch/opt.py salad_torch/modules/fast.py
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# T: per-shape cuBLAS vs Triton template. One LENGTH per process, one PRECISION
# per process (the precision is a process-global that inductor reads at compile).
for L in 100 400; do
  for P in highest high; do
    N020_LEN=$L N020_PREC=$P TORCHINDUCTOR_CACHE_DIR=/tmp/n020t_${SLURM_JOB_ID}_${L}_${P} \
      python -u $R/x_pershape.py || true
  done
done
rm -rf /tmp/n020t_${SLURM_JOB_ID}_*
