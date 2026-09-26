#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --job-name=n020_bench
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering/gpu_bench_%j.log
# PLACEMENT: 1x H100, 16 cores. GPU-only (it is a CUDA-graph replay benchmark).
# ONE length and ONE arm per process - several shapes in one process degrades
# cudagraphs and once produced a false "flat 17.7 ms" (N009/N012).
# ~4 GPU-h budgeted. Autotune prints "AUTOTUNE mm(...)" per shape: that log IS
# the per-shape selection table.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n020
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCHINDUCTOR_PRINT_AUTOTUNE=1
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering
cd $SALAD_TREE
md5sum salad_torch/modules/basic.py salad_torch/opt.py salad_torch/modules/fast.py
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
for L in 100 400; do
  for ARM in "base einsum 0" "mm mm 0" "mmtpl mm 1" "addmmtpl addmm 1"; do
    set -- $ARM; TAG=$1; MODE=$2; TPL=$3
    echo "########## ARM=$TAG L=$L"
    N020_TAG=${TAG}_L${L} N020_LEN=$L SALAD_OPT_MM=$MODE SALAD_OPT_GEMMTPL=$TPL \
      N020_REPS=3 TORCHINDUCTOR_CACHE_DIR=/tmp/n020b_${SLURM_JOB_ID}_${TAG}_${L} \
      python -u $R/x_bench.py || true
  done
done
rm -rf /tmp/n020b_${SLURM_JOB_ID}_*
