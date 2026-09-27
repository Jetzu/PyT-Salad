#!/bin/bash -l
#SBATCH --job-name=n028_ctrl2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor/logs/ctrl2_%j.log
set -u
# PLACEMENT: CPU, 48 cores (full NUMA fraction). This is the BITWISE control -
# CPU is deterministic by construction, so it is the only place a bit-exactness
# claim can actually be made, and it needs no GPU at all.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export OMP_NUM_THREADS=48
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
export SALAD_TREE=/home/hpc/v181bb/v181bb10/n028wt_o8
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor
cd $SALAD_TREE
md5sum salad_torch/modules/blocks.py salad_torch/geometry/features.py salad_torch/geometry/frames.py salad_torch/geometry/rigid.py salad_torch/opt.py
D=${TMPDIR:-/tmp}/n028bits_${SLURM_JOB_ID}
mkdir -p $D
for L in 64 100; do
  for arm in base o6 o8 o6o8; do
    case $arm in
      base) IPA=0; PFM=0; FMS=0;;
      o6)   IPA=0; PFM=1; FMS=0;;
      o8)   IPA=0; PFM=0; FMS=1;;
      o6o8) IPA=0; PFM=1; FMS=1;;
    esac
    SALAD_OPT_IPAMULSUM=$IPA SALAD_OPT_PAIRFEAT=$PFM SALAD_OPT_FRAMEMULSUM=$FMS \
    N028_DUMP=$D/${arm}_$L N028_LEN=$L \
    TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/n028cc_${SLURM_JOB_ID}_${arm}_$L \
      $P -u $R/x_cpu_bits.py
    rm -rf ${TMPDIR:-/tmp}/n028cc_${SLURM_JOB_ID}_${arm}_$L
  done
  for arm in o6 o8 o6o8; do
    echo "===== L=$L  $arm vs base"
    $P $R/x_cmp_bits.py $D/base_$L $D/${arm}_$L "L=$L $arm" || true
  done
done
rm -rf $D
