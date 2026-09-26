#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=n020_stab
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering/gpu_stability_%j.log
# PLACEMENT: 1x H100, 16 cores. GPU-only. ~1 GPU-h.
# THE QUESTION THIS ANSWERS: with `max_autotune_gemm` the kernel choice is a
# BENCHMARK, so two independent compiles with a COLD autotune cache could in
# principle disagree at a shape where the candidates are within noise - which
# would make the preset's bits depend on a timing race, not just on the device.
# Three independent cold-cache processes, same length, compared bitwise under
# DETSEG. Plus a 4th arm with templates OFF as the reference.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n020
export SALAD_OPT_DETSEG=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCHINDUCTOR_PRINT_AUTOTUNE=1
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering
cd $SALAD_TREE
L=100
for i in 1 2 3; do
  echo "########## cold-cache tpl1 replicate $i"
  SALAD_OPT_MM=mm SALAD_OPT_GEMMTPL=1 N020_LEN=$L N020_DUMP=/tmp/n020s_r$i.pt \
    TORCHINDUCTOR_CACHE_DIR=/tmp/n020sc_${SLURM_JOB_ID}_$i python -u $R/x_gpu_bits.py || true
done
echo "########## templates OFF reference"
SALAD_OPT_MM=mm SALAD_OPT_GEMMTPL=0 N020_LEN=$L N020_DUMP=/tmp/n020s_off.pt \
  TORCHINDUCTOR_CACHE_DIR=/tmp/n020sc_${SLURM_JOB_ID}_off python -u $R/x_gpu_bits.py || true
for t in r2 r3 off; do
  python -u $R/x_cmp.py "tpl1 r1 vs $t" /tmp/n020s_r1.pt /tmp/n020s_$t.pt
done
rm -f /tmp/n020s_*.pt; rm -rf /tmp/n020sc_${SLURM_JOB_ID}_*
