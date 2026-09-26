#!/bin/bash -l
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --job-name=n020_cpu
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering/cpu_%j.log
# PLACEMENT: cpu partition, 48 cores (full NUMA fraction). No GPU: the shape
# census is device-independent and the bitwise control REQUIRES CPU (it is
# deterministic by construction there; on CUDA the segment reduction's atomicAdd
# is not, unless DETSEG - see gpu_bits).
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n020
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering
cd $SALAD_TREE
md5sum salad_torch/modules/basic.py salad_torch/opt.py salad_torch/modules/fast.py
echo "########## S: shape census"
for L in 100 400; do
  for AA in 0 1; do
    N020_LEN=$L N020_NEEDAA=$AA python -u $R/x_shapes.py
  done
done
echo "########## C: CPU bitwise control, compiled forward_core, one arm per process"
for L in 64 100; do
  for MODE in einsum mm addmm; do
    SALAD_OPT_MM=$MODE N020_LEN=$L N020_DUMP=/tmp/n020_${L}_${MODE}.pt \
      TORCHINDUCTOR_CACHE_DIR=/tmp/n020cc_${SLURM_JOB_ID}_${L}_${MODE} \
      python -u $R/x_cpu_bits.py
  done
  for MODE in mm addmm; do
    python -u $R/x_cmp.py "L=$L $MODE vs einsum" /tmp/n020_${L}_einsum.pt /tmp/n020_${L}_${MODE}.pt
  done
done
rm -f /tmp/n020_*.pt; rm -rf /tmp/n020cc_${SLURM_JOB_ID}_*
