#!/bin/bash -l
#SBATCH --job-name=n028_b2
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor/logs/bench2_%j.log
set -u
# PLACEMENT: 1x H100, captured-graph replay timing. ~2 GPU-h worst case.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
export SALAD_TREE=/home/hpc/v181bb/v181bb10/n028wt_o8
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
cd $SALAD_TREE
git -C $SALAD_TREE rev-parse HEAD
md5sum salad_torch/geometry/frames.py salad_torch/geometry/rigid.py \
       salad_torch/geometry/features.py salad_torch/modules/blocks.py salad_torch/opt.py
nvidia-smi --query-gpu=name --format=csv,noheader
for rep in 1 2; do
  for arm in base o8 o6o8 o2o6o8; do
    case $arm in
      base)    IPA=0; PFM=0; FMS=0;;
      o8)      IPA=0; PFM=0; FMS=1;;
      o6o8)    IPA=0; PFM=1; FMS=1;;
      o2o6o8)  IPA=1; PFM=1; FMS=1;;
    esac
    SALAD_OPT_IPAMULSUM=$IPA SALAD_OPT_PAIRFEAT=$PFM SALAD_OPT_FRAMEMULSUM=$FMS \
    N028_TAG=${arm}_L100_r${rep} N028_LEN=100 \
    TORCHINDUCTOR_CACHE_DIR=/dev/shm/n028b2_${SLURM_JOB_ID}_${arm} \
      $P -u $R/x_bench.py || true
    rm -rf /dev/shm/n028b2_${SLURM_JOB_ID}_${arm}
  done
done
# length sweep for the best bit-exact-or-not combination
for L in 50 200 400; do
  for arm in base o6o8; do
    case $arm in base) IPA=0; PFM=0; FMS=0;; o6o8) IPA=0; PFM=1; FMS=1;; esac
    SALAD_OPT_IPAMULSUM=$IPA SALAD_OPT_PAIRFEAT=$PFM SALAD_OPT_FRAMEMULSUM=$FMS \
    N028_TAG=${arm}_L${L}_sweep N028_LEN=$L \
    TORCHINDUCTOR_CACHE_DIR=/dev/shm/n028b2_${SLURM_JOB_ID}_${arm}_${L} \
      $P -u $R/x_bench.py || true
    rm -rf /dev/shm/n028b2_${SLURM_JOB_ID}_${arm}_${L}
  done
done
