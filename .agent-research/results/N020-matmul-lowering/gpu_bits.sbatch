#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=n020_gpubits
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering/gpu_bits_%j.log
# PLACEMENT: 1x H100, 16 cores. GPU-only: this control exists precisely to test
# the kernel SELECTION that only happens on a GPU. Under SALAD_OPT_DETSEG=1 the
# comparison is bitwise (N013/N019 method).
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n020
export SALAD_OPT_DETSEG=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering
cd $SALAD_TREE
md5sum salad_torch/modules/basic.py salad_torch/opt.py
for L in 100 400; do
  # arm name : SALAD_OPT_MM / SALAD_OPT_GEMMTPL
  for ARM in "einsum 0" "einsum 0 rerun" "mm 0" "mm 1" "addmm 1"; do
    set -- $ARM; MODE=$1; TPL=$2; SUF=${3:-}
    NAME=${MODE}_tpl${TPL}${SUF:+_$SUF}
    SALAD_OPT_MM=$MODE SALAD_OPT_GEMMTPL=$TPL N020_LEN=$L \
      N020_DUMP=/tmp/n020g_${L}_${NAME}.pt \
      TORCHINDUCTOR_CACHE_DIR=/tmp/n020gc_${SLURM_JOB_ID}_${L}_${NAME} \
      python -u $R/x_gpu_bits.py || true
  done
  echo "##### L=$L comparisons (base = einsum_tpl0)"
  for NAME in einsum_tpl0_rerun mm_tpl0 mm_tpl1 addmm_tpl1; do
    [ -f /tmp/n020g_${L}_${NAME}.pt ] && \
      python -u $R/x_cmp.py "L=$L $NAME vs einsum" /tmp/n020g_${L}_einsum_tpl0.pt /tmp/n020g_${L}_${NAME}.pt
  done
done
rm -f /tmp/n020g_*.pt; rm -rf /tmp/n020gc_${SLURM_JOB_ID}_*
