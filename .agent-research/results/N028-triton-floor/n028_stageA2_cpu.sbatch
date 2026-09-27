#!/bin/bash -l
#SBATCH --job-name=n028A2_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH -o /home/hpc/v181bb/v181bb10/salad_jobs/n028_stageA2_cpu_%j.log
set -u
# PLACEMENT: CPU, 48 cores (full NUMA fraction, matches p7final_cpu.sbatch).
# RE-RUN after fixing O6: the naive merge (sharing the batched (2A) transform)
# was compiled-bit-exact but NOT eager-bit-exact - a live-process-only ~1e-7
# perturbation from block 2 onward that test_sampling_parity's chaos amplified
# to 7/64 (FAIL). Fix: share only aa_frames_t + the neighbour gather (both
# rounding-free); call each function's OWN apply_inverse_t/vec_normalized
# unchanged in shape and call count. This job re-verifies: CPU bitwise control
# (compiled forward_core) + the full parity suite in the shipped default state.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export OMP_NUM_THREADS=48
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor
SRC=/home/hpc/v181bb/v181bb10/SALAD_torch
W=${TMPDIR:-/tmp}/n028A2wt_${SLURM_JOB_ID}
rm -rf $W; mkdir -p $W
cp -r $SRC/. $W/ 2>/dev/null
cd $W
echo "frozen from HEAD: $(git -C $SRC rev-parse HEAD) (uncommitted diff below)"
git -C $SRC diff --stat HEAD -- salad_torch/ | tail -20
export SALAD_TREE=$W
D=${TMPDIR:-/tmp}/n028A2bits_${SLURM_JOB_ID}
mkdir -p $D
echo "############### O6 (FIXED) CPU BITWISE CONTROL - compiled forward_core"
for L in 64 100; do
  for arm in base o6; do
    case $arm in
      base) PFM=0;;
      o6)   PFM=1;;
    esac
    SALAD_OPT_PAIRFEAT=$PFM \
    N028_DUMP=$D/${arm}_$L N028_LEN=$L \
    TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/n028A2cc_${SLURM_JOB_ID}_${arm}_$L \
      $P -u $R/x_cpu_bits.py
    rm -rf ${TMPDIR:-/tmp}/n028A2cc_${SLURM_JOB_ID}_${arm}_$L
  done
  echo "===== L=$L  O6 (PAIR_FEAT_MERGE) vs base  -- must be bit_identical=True"
  $P $R/x_cmp_bits.py $D/base_$L $D/o6_$L "L=$L o6"
done
rm -rf $D

echo "############### O6 (FIXED) EAGER CPU CONTROL - test_sampling_parity, both flag states"
for pfm in 0 1; do
  echo "----- SALAD_OPT_PAIRFEAT=$pfm"
  SALAD_OPT_PAIRFEAT=$pfm $P tests/test_sampling_parity.py 2>&1 | grep -E "final_aatype|worst/noise-floor ratio|final/noise-floor|^OK|^FAILED"
done

echo "############### PARITY SUITE, default state (PAIR_FEAT_MERGE=True baked in)"
rc=0
for cfg in default_vp default_vp_scaled default_ve_scaled multimotif_vp; do
  echo "===== test_model_parity $cfg"; SALAD_CONFIG=$cfg $P tests/test_model_parity.py; e=$?; echo "EXIT $e : test_model_parity $cfg"; [ $e -ne 0 ] && rc=1
done
for t in geometry stack encoder sampling; do
  echo "===== test_${t}_parity"; $P tests/test_${t}_parity.py; e=$?; echo "EXIT $e : test_${t}_parity"; [ $e -ne 0 ] && rc=1
done
echo "CPU_SUITE_EXIT=$rc"
rm -rf $W
