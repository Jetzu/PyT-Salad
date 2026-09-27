#!/bin/bash -l
#SBATCH --job-name=n028_bench
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor/logs/bench_%j.log
set -u
# PLACEMENT: 1x H100. Timing the CAPTURED CUDA-graph replay - GPU-only by
# definition. ~2 GPU-h worst case.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n028wt
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
cd $SALAD_TREE
git -C $SALAD_TREE rev-parse HEAD
md5sum salad_torch/modules/attention.py salad_torch/modules/blocks.py salad_torch/geometry/features.py salad_torch/opt.py
nvidia-smi --query-gpu=name --format=csv,noheader

# 2 reps x 4 arms x 1 length, ONE PROCESS EACH (several shapes/configs in one
# process degrades cudagraphs - N009/N012 method note), own inductor cache per
# arm so no graph can leak between arms. Caches on /dev/shm: /hnvme is
# inode-capped and in grace, and one length's cache is ~3,800 files.
for rep in 1 2; do
for L in 100; do
  for arm in base o2 o6 o2o6; do
    case $arm in
      base) IPA=0; PFM=0;;
      o2)   IPA=1; PFM=0;;
      o6)   IPA=0; PFM=1;;
      o2o6) IPA=1; PFM=1;;
    esac
    SALAD_OPT_IPAMULSUM=$IPA SALAD_OPT_PAIRFEAT=$PFM \
    N028_TAG=${arm}_L${L}_r${rep} N028_LEN=$L \
    TORCHINDUCTOR_CACHE_DIR=/dev/shm/n028b_${SLURM_JOB_ID}_${arm}_${L} \
      $P -u $R/x_bench.py || true
    rm -rf /dev/shm/n028b_${SLURM_JOB_ID}_${arm}_${L}
  done
done
done

# the winner's length sweep (one process per length, as above)
for L in 50 200 400; do
  for arm in base o6; do
    case $arm in base) IPA=0; PFM=0;; o6) IPA=0; PFM=1;; esac
    SALAD_OPT_IPAMULSUM=$IPA SALAD_OPT_PAIRFEAT=$PFM \
    N028_TAG=${arm}_L${L}_sweep N028_LEN=$L \
    TORCHINDUCTOR_CACHE_DIR=/dev/shm/n028b_${SLURM_JOB_ID}_${arm}_${L} \
      $P -u $R/x_bench.py || true
    rm -rf /dev/shm/n028b_${SLURM_JOB_ID}_${arm}_${L}
  done
done
