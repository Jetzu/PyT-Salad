#!/bin/bash -l
#SBATCH --job-name=n028_b3
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor/logs/bench3_%j.log
set -u
# PLACEMENT: 1x H100. The `fast` preset (TF32) is where O2+O8 actually ship, and
# TF32 makes the GEMMs cheaper, so the -24.5 % measured on `exact` is not
# automatically the `fast` number. <1 GPU-h.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
export SALAD_TREE=/home/hpc/v181bb/v181bb10/n028wt_o8
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
cd $SALAD_TREE
for rep in 1 2; do
for arm in base o6 all; do
  case $arm in
    base) IPA=0; PFM=0; FMS=0;;
    o6)   IPA=0; PFM=1; FMS=0;;
    all)  IPA=1; PFM=1; FMS=1;;
  esac
  SALAD_OPT_IPAMULSUM=$IPA SALAD_OPT_PAIRFEAT=$PFM SALAD_OPT_FRAMEMULSUM=$FMS \
  N028_TAG=fast_${arm}_r${rep} N028_LEN=100 N028_PRESET=fast \
  TORCHINDUCTOR_CACHE_DIR=/dev/shm/n028b3_${SLURM_JOB_ID}_${arm} \
    $P -u $R/x_bench.py || true
  rm -rf /dev/shm/n028b3_${SLURM_JOB_ID}_${arm}
done
done
