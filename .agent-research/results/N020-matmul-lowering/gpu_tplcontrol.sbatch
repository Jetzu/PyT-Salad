#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=n020_gputpl
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering/gpu_tplcontrol_%j.log
# PLACEMENT: 1x H100, 16 cores. GPU-only, ~1.5 GPU-h budgeted (2 lengths x
# ~12 min template compile x margin).
# THE DECIDING CONTROL for this item: the prior agent's gpu_bits.sbatch varied
# SALAD_OPT_MM x SALAD_OPT_GEMMTPL but never tested einsum+tpl1 bitwise against
# einsum+tpl0 - and SALAD_OPT_MM defaults to einsum (this item restores that
# default; mm/addmm are rejected, see RESULTS.md section on MM_MODE). So the
# combination that actually SHIPS if templates land was never checked bitwise.
# Under SALAD_OPT_DETSEG=1 the only source of CUDA run-to-run nondeterminism
# (the segment reduction's atomicAdd) is gone, so this comparison is bitwise
# (N013/N019/N020's own gpu_bits.sbatch method).
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n020
export SALAD_OPT_DETSEG=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering
cd $SALAD_TREE
md5sum salad_torch/modules/basic.py salad_torch/opt.py
for L in 100 400; do
  for TPL in 0 1; do
    SALAD_OPT_MM=einsum SALAD_OPT_GEMMTPL=$TPL N020_LEN=$L \
      N020_DUMP=/tmp/n020gt_${L}_tpl${TPL}.pt \
      TORCHINDUCTOR_CACHE_DIR=/tmp/n020gtc_${SLURM_JOB_ID}_${L}_${TPL} \
      python -u $R/x_gpu_bits.py || true
  done
  echo "##### L=$L einsum tpl1 vs tpl0 (bitwise, DETSEG)"
  [ -f /tmp/n020gt_${L}_tpl1.pt ] && [ -f /tmp/n020gt_${L}_tpl0.pt ] && \
    python -u $R/x_cmp.py "L=$L einsum tpl1 vs tpl0" \
      /tmp/n020gt_${L}_tpl0.pt /tmp/n020gt_${L}_tpl1.pt
done
rm -f /tmp/n020gt_*.pt; rm -rf /tmp/n020gtc_${SLURM_JOB_ID}_*
