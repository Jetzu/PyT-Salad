#!/bin/bash -l
#SBATCH --job-name=n028_gbits
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor/logs/gbits_%j.log
set -u
# PLACEMENT: 1x H100. The second half of `aggressive_fusion`'s admission
# standard (N013): bit-identical on the GPU under DETSEG, where there is no
# atomics noise left to hide behind. <1 GPU-h.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
export SALAD_TREE=/home/hpc/v181bb/v181bb10/n028wt_o8
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor
cd $SALAD_TREE
D=/dev/shm/n028gb_${SLURM_JOB_ID}; mkdir -p $D
for L in 64 100; do
  for arm in base baserep o6 o8 o6o8; do
    case $arm in
      base|baserep) IPA=0; PFM=0; FMS=0;;
      o6)   IPA=0; PFM=1; FMS=0;;
      o8)   IPA=0; PFM=0; FMS=1;;
      o6o8) IPA=0; PFM=1; FMS=1;;
    esac
    SALAD_OPT_IPAMULSUM=$IPA SALAD_OPT_PAIRFEAT=$PFM SALAD_OPT_FRAMEMULSUM=$FMS \
    N028_DUMP=$D/${arm}_$L N028_LEN=$L \
    TORCHINDUCTOR_CACHE_DIR=/dev/shm/n028gbc_${SLURM_JOB_ID}_${arm}_$L \
      $P -u $R/x_gpubits.py
    rm -rf /dev/shm/n028gbc_${SLURM_JOB_ID}_${arm}_$L
  done
  for arm in baserep o6 o8 o6o8; do
    echo "===== GPU/DETSEG L=$L  $arm vs base  (baserep is the self-floor)"
    $P $R/x_cmp_bits.py $D/base_$L $D/${arm}_$L "GPU L=$L $arm" || true
  done
done
rm -rf $D
