#!/bin/bash -l
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --job-name=n020_cpudiag
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering/cpu_diag_%j.log
# PLACEMENT: cpu partition, 48 cores. No GPU: this diagnoses the COMPILED-CPU
# delta the mandated bitwise control found (mm vs einsum, 5.6e-05 on pos at
# L=64) - in isolation, per shape, compiled bmm and compiled mm are bit-equal on
# CPU at all seven of the model's shapes, so the delta must come from inductor's
# FUSION decisions, not from the BLAS entry point.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n020
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N020-matmul-lowering
cd $SALAD_TREE
L=64
# A: plain compile, no CSE, no aggressive_fusion  -> is the delta fusion-driven?
for MODE in einsum mm; do
  SALAD_OPT_MM=$MODE N020_LEN=$L N020_CSE=0 N020_AGGR=0 N020_DUMP=/tmp/n020d_plain_${MODE}.pt \
    TORCHINDUCTOR_CACHE_DIR=/tmp/n020dc_${SLURM_JOB_ID}_p_${MODE} python -u $R/x_cpu_bits.py
done
python -u $R/x_cmp.py "PLAIN(no cse,no aggr) mm vs einsum" /tmp/n020d_plain_einsum.pt /tmp/n020d_plain_mm.pt
# B: CSE only
for MODE in einsum mm; do
  SALAD_OPT_MM=$MODE N020_LEN=$L N020_CSE=1 N020_AGGR=0 N020_DUMP=/tmp/n020d_cse_${MODE}.pt \
    TORCHINDUCTOR_CACHE_DIR=/tmp/n020dc_${SLURM_JOB_ID}_c_${MODE} python -u $R/x_cpu_bits.py
done
python -u $R/x_cmp.py "CSE only mm vs einsum" /tmp/n020d_cse_einsum.pt /tmp/n020d_cse_mm.pt
# C: aggressive_fusion only
for MODE in einsum mm; do
  SALAD_OPT_MM=$MODE N020_LEN=$L N020_CSE=0 N020_AGGR=1 N020_DUMP=/tmp/n020d_aggr_${MODE}.pt \
    TORCHINDUCTOR_CACHE_DIR=/tmp/n020dc_${SLURM_JOB_ID}_a_${MODE} python -u $R/x_cpu_bits.py
done
python -u $R/x_cmp.py "AGGR only mm vs einsum" /tmp/n020d_aggr_einsum.pt /tmp/n020d_aggr_mm.pt
# D: generated code - which extern kernels / codegen'd kernels does each arm use?
export TORCH_COMPILE_DEBUG=1
for MODE in einsum mm; do
  TORCH_COMPILE_DEBUG_DIR=/tmp/n020dbg_${SLURM_JOB_ID}_${MODE} \
    SALAD_OPT_MM=$MODE N020_LEN=$L N020_CSE=0 N020_AGGR=0 N020_DUMP=/tmp/n020d_dbg_${MODE}.pt \
    TORCHINDUCTOR_CACHE_DIR=/tmp/n020dc_${SLURM_JOB_ID}_d_${MODE} python -u $R/x_cpu_bits.py
  F=$(find /tmp/n020dbg_${SLURM_JOB_ID}_${MODE} -name output_code.py | head -1)
  echo "##### $MODE output_code=$F"
  echo "  extern mm   : $(grep -c 'extern_kernels.mm' $F)"
  echo "  extern bmm  : $(grep -c 'extern_kernels.bmm' $F)"
  echo "  extern addmm: $(grep -c 'extern_kernels.addmm' $F)"
  echo "  cpp kernels : $(grep -c 'async_compile.cpp' $F)"
  grep -oE 'extern_kernels\.[a-z_]+' $F | sort | uniq -c
  gzip -c $F > $R/cpu_output_code_${MODE}.py.gz
done
rm -f /tmp/n020d_*.pt; rm -rf /tmp/n020dc_${SLURM_JOB_ID}_* /tmp/n020dbg_${SLURM_JOB_ID}_*
