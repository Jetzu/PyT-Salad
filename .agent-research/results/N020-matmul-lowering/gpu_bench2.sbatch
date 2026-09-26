#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=n020_bench2
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering/gpu_bench2_%j.log
# PLACEMENT: 1x H100, 16 cores. GPU-only. ~2 GPU-h.
# CONTROL for the claim "batch-1 bmm forecloses inductor's matmul templates":
# inductor DOES have a `tuned_bmm`, so turning templates on WITHOUT the mm
# rewrite is the arm that separates "templates" from "mm lowering".
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n020
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCHINDUCTOR_PRINT_AUTOTUNE=1
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering
cd $SALAD_TREE
for L in 100 400; do
  echo "########## ARM=einsumtpl L=$L"
  N020_TAG=einsumtpl_L${L} N020_LEN=$L SALAD_OPT_MM=einsum SALAD_OPT_GEMMTPL=1 \
    N020_REPS=3 TORCHINDUCTOR_CACHE_DIR=/tmp/n020e_${SLURM_JOB_ID}_${L} \
    python -u $R/x_bench.py || true
done
rm -rf /tmp/n020e_${SLURM_JOB_ID}_*
