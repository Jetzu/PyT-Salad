#!/bin/bash -l
#SBATCH --job-name=n028B_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH -o /home/hpc/v181bb/v181bb10/salad_jobs/n028_stageB_cpu_%j.log
set -u
# PLACEMENT: CPU, 48 cores. (1) bitwise control for O8 (FRAME_MULSUM) on top
# of the now-default O6; (2) parity suite in the SHIPPED default state
# (PAIR_FEAT_MERGE=True, FRAME_MULSUM=False outside `fast` - no env override,
# same p7final_cpu.sbatch pattern).
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export OMP_NUM_THREADS=48
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor
SRC=/home/hpc/v181bb/v181bb10/SALAD_torch
W=${TMPDIR:-/tmp}/n028Bwt_${SLURM_JOB_ID}
rm -rf $W; mkdir -p $W
cp -r $SRC/. $W/ 2>/dev/null
cd $W
echo "frozen from HEAD: $(git -C $SRC rev-parse HEAD) (uncommitted diff below)"
git -C $SRC diff --stat HEAD -- salad_torch/ | tail -30
export SALAD_TREE=$W
D=${TMPDIR:-/tmp}/n028Bbits_${SLURM_JOB_ID}
mkdir -p $D
echo "############### O8 CPU BITWISE CONTROL (on top of default O6)"
for L in 64 100; do
  for arm in base o8; do
    case $arm in
      base) FMS=0;;
      o8)   FMS=1;;
    esac
    SALAD_OPT_FRAMEMULSUM=$FMS \
    N028_DUMP=$D/${arm}_$L N028_LEN=$L \
    TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/tmp}/n028Bcc_${SLURM_JOB_ID}_${arm}_$L \
      $P -u $R/x_cpu_bits.py
    rm -rf ${TMPDIR:-/tmp}/n028Bcc_${SLURM_JOB_ID}_${arm}_$L
  done
  echo "===== L=$L  O8 (FRAME_MULSUM) vs base  -- reassociation, checking whether it happens to be bit-exact"
  $P $R/x_cmp_bits.py $D/base_$L $D/o8_$L "L=$L o8" || true
done
rm -rf $D

echo "############### PARITY SUITE, default state (frame_mulsum only True under --preset fast)"
rc=0
for cfg in default_vp default_vp_scaled default_ve_scaled multimotif_vp; do
  echo "===== test_model_parity $cfg"; SALAD_CONFIG=$cfg $P tests/test_model_parity.py; e=$?; echo "EXIT $e : test_model_parity $cfg"; [ $e -ne 0 ] && rc=1
done
for t in geometry stack encoder sampling; do
  echo "===== test_${t}_parity"; $P tests/test_${t}_parity.py; e=$?; echo "EXIT $e : test_${t}_parity"; [ $e -ne 0 ] && rc=1
done
echo "CPU_SUITE_EXIT=$rc"
rm -rf $W
