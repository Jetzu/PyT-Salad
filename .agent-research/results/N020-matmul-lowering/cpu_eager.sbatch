#!/bin/bash -l
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --job-name=n020_cpueager
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering/cpu_eager_%j.log
# PLACEMENT: cpu partition, 48 cores, no GPU.
# THE 2x2 the first control was missing: {eager, compiled} x {einsum, mm}, same
# process count, same thread count. Eager einsum == eager mm was already shown
# (the whole parity gate is byte-identical), and compiled einsum != compiled mm
# by 4.2e-05 on pos. This says WHICH of the two compiled arms moved.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n020
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering
cd $SALAD_TREE
for L in 64 100; do
  for MODE in einsum mm addmm; do
    SALAD_OPT_MM=$MODE N020_EAGER=1 N020_LEN=$L N020_DUMP=/tmp/n020e_${L}_eager_${MODE}.pt \
      python -u $R/x_cpu_bits.py
    SALAD_OPT_MM=$MODE N020_LEN=$L N020_DUMP=/tmp/n020e_${L}_cmp_${MODE}.pt \
      TORCHINDUCTOR_CACHE_DIR=/tmp/n020ec_${SLURM_JOB_ID}_${L}_${MODE} python -u $R/x_cpu_bits.py
  done
  for MODE in mm addmm; do
    python -u $R/x_cmp.py "L=$L EAGER $MODE vs einsum" /tmp/n020e_${L}_eager_einsum.pt /tmp/n020e_${L}_eager_${MODE}.pt
  done
  for MODE in einsum mm addmm; do
    python -u $R/x_cmp.py "L=$L $MODE: compiled vs eager" /tmp/n020e_${L}_eager_${MODE}.pt /tmp/n020e_${L}_cmp_${MODE}.pt
  done
done
rm -f /tmp/n020e_*.pt; rm -rf /tmp/n020ec_${SLURM_JOB_ID}_*
