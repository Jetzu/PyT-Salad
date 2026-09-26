#!/bin/bash -l
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --job-name=n020_cputpl
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering/cpu_tplcontrol_%j.log
# PLACEMENT: cpu partition, 48 cores. No GPU: deliverable-1's CPU leg, testing
# GEMM_TEMPLATES specifically (the earlier cpu/cpu_diag/cpu_eager jobs only
# ever varied SALAD_OPT_MM, never SALAD_OPT_GEMMTPL - a gap in the prior
# agent's run). SALAD_OPT_MM=einsum throughout because that is the SHIPPED
# default (bit-exact form) this item restores; this is the exact combination
# that ships if templates land. Expect a no-op on CPU (Triton templates are a
# CUDA-only inductor backend) - this job exists to CONFIRM that, not assume it.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n020
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering
cd $SALAD_TREE
md5sum salad_torch/modules/basic.py salad_torch/opt.py
for L in 64 100; do
  for TPL in 0 1; do
    SALAD_OPT_MM=einsum SALAD_OPT_GEMMTPL=$TPL N020_LEN=$L \
      N020_DUMP=/tmp/n020ct_${L}_tpl${TPL}.pt \
      TORCHINDUCTOR_CACHE_DIR=/tmp/n020ctc_${SLURM_JOB_ID}_${L}_${TPL} \
      python -u $R/x_cpu_bits.py
  done
  python -u $R/x_cmp.py "L=$L einsum tpl1 vs tpl0 (CPU)" \
    /tmp/n020ct_${L}_tpl0.pt /tmp/n020ct_${L}_tpl1.pt
done
rm -f /tmp/n020ct_*.pt; rm -rf /tmp/n020ctc_${SLURM_JOB_ID}_*
