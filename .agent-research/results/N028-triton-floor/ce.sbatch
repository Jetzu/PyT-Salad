#!/bin/bash -l
#SBATCH --job-name=n028_ce
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor/logs/ce_%j.log
set -u
# PLACEMENT: 1x H100. K021's missing check - COMPILED vs EAGER whole-forward on
# the GPU, which is the only thing that could have seen P7's miscompile.
# The flags-off arm in the SAME job is the floor. <1 GPU-h.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
export SALAD_TREE=/home/hpc/v181bb/v181bb10/n028wt_o8
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor
cd $SALAD_TREE
md5sum salad_torch/geometry/frames.py salad_torch/geometry/features.py salad_torch/modules/attention.py
for L in 64 100; do
for arm in floor o6 o8 o2 o6o8 all; do
  case $arm in
    floor) IPA=0; PFM=0; FMS=0;;
    o6)    IPA=0; PFM=1; FMS=0;;
    o8)    IPA=0; PFM=0; FMS=1;;
    o2)    IPA=1; PFM=0; FMS=0;;
    o6o8)  IPA=0; PFM=1; FMS=1;;
    all)   IPA=1; PFM=1; FMS=1;;
  esac
  SALAD_OPT_IPAMULSUM=$IPA SALAD_OPT_PAIRFEAT=$PFM SALAD_OPT_FRAMEMULSUM=$FMS \
  N028_TAG=${arm}_L$L N028_LEN=$L \
  TORCHINDUCTOR_CACHE_DIR=/dev/shm/n028ce_${SLURM_JOB_ID}_${arm}_$L \
    $P -u $R/x_ce.py || true
  rm -rf /dev/shm/n028ce_${SLURM_JOB_ID}_${arm}_$L
done
done
