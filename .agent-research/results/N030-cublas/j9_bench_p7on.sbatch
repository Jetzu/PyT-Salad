#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=n030_p7on
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas/j9_bench_p7on_%j.log
# PLACEMENT: 1x H100, 16 cores, <2 GPU-h. Re-measure decomp32's MARGINAL
# contribution on top of the CURRENT shipped baseline (P7 re-enabled via the
# 40a85be fusion barrier, opt.BATCH_COND defaults True) - the earlier -16.6%
# number was measured pre-P7-reenable and may not hold now that P7 already
# removes some of the same 3x3-frame-rotation GEMMs. One (arm) per process.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas
cd $R
git -C /home/hpc/v181bb/v181bb10/SALAD_torch rev-parse HEAD
md5sum /hnvme/workspace/v181bb10-salad/n030/salad_torch/gemm_exact.py \
       /hnvme/workspace/v181bb10-salad/n030/salad_torch/modules/fast.py \
       /hnvme/workspace/v181bb10-salad/n030/salad_torch/opt.py \
       /hnvme/workspace/v181bb10-salad/n030/salad_torch/modules/cond_prologue.py
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
for ARM in off decomp32; do
  echo "########## ARM=$ARM L=100"
  TMPD=/dev/shm/n030p7_${SLURM_JOB_ID}_${ARM}
  mkdir -p $TMPD
  SALAD_TREE=/hnvme/workspace/v181bb10-salad/n030 N030_TAG="${ARM}_L100" \
    N030_LEN=100 N030_PRESET=exact N030_ARM=$ARM N030_REPS=3 \
    TORCHINDUCTOR_CACHE_DIR=$TMPD python -u $R/x_bench.py
  rm -rf $TMPD
done
