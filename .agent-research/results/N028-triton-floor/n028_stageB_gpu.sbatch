#!/bin/bash -l
#SBATCH --job-name=n028B_gpu
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /home/hpc/v181bb/v181bb10/salad_jobs/n028_stageB_gpu_%j.log
set -u
# PLACEMENT: 1x H100. Parity suite (default state) + n=160 design-quality gate
# for O8 forced on via env var over the `fast` preset (OR contract - identical
# to shipping frame_mulsum=True in the preset dict, which is flipped only
# after this passes) + n=64 exact quality + speed re-baseline. ~1.5 GPU-h.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind
SRC=/home/hpc/v181bb/v181bb10/SALAD_torch
W=/dev/shm/n028Bwt_${SLURM_JOB_ID}
rm -rf $W; mkdir -p $W
cp -r $SRC/. $W/
cd $W
echo "frozen from HEAD: $(git -C $SRC rev-parse HEAD)"
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
nvidia-smi --query-gpu=name --format=csv,noheader

echo "############### PARITY SUITE, default state"
rc=0
for cfg in default_vp default_vp_scaled default_ve_scaled multimotif_vp; do
  echo "===== test_model_parity $cfg"; SALAD_CONFIG=$cfg $P tests/test_model_parity.py; e=$?; echo "EXIT $e : test_model_parity $cfg"; [ $e -ne 0 ] && rc=1
done
for t in geometry stack encoder sampling; do
  echo "===== test_${t}_parity"; $P tests/test_${t}_parity.py; e=$?; echo "EXIT $e : test_${t}_parity"; [ $e -ne 0 ] && rc=1
done

echo "############### n=160 DESIGN QUALITY: base vs O8 (env-forced), fast preset"
OUT=/dev/shm/n028Bdesigns_${SLURM_JOB_ID}
mkdir -p $OUT
for arm in base o8; do
  case $arm in
    base) export SALAD_OPT_FRAMEMULSUM=0;;
    o8)   export SALAD_OPT_FRAMEMULSUM=1;;
  esac
  for spec in "42 40" "1234 120"; do
    set -- $spec; SEED=$1; N=$2
    D=$OUT/${arm}_s${SEED}
    rm -rf $D
    export TORCHINDUCTOR_CACHE_DIR=/dev/shm/n028Bqc_${SLURM_JOB_ID}_${arm}
    echo "########## ARM $arm seed $SEED n $N (FRAMEMULSUM=$SALAD_OPT_FRAMEMULSUM)"
    $P tests/test_design_quality.py --n $N --preset fast --seed $SEED --length 100 --out_path $D --keep
    e=$?; echo "GATE_EXIT arm=$arm seed=$SEED rc=$e"; [ $e -ne 0 ] && rc=1
  done
  rm -rf /dev/shm/n028Bqc_${SLURM_JOB_ID}_${arm}
done
unset SALAD_OPT_FRAMEMULSUM
echo "===== quality n=64 exact"; $P tests/test_design_quality.py --n 64 --preset exact --out_path ${TMPDIR:-/dev/shm}/f2_${SLURM_JOB_ID}; e=$?; echo "EXIT $e : quality exact n=64"; [ $e -ne 0 ] && rc=1
rm -rf ${TMPDIR:-/dev/shm}/f2_${SLURM_JOB_ID}

echo "############### POOLED n=160 STATS (N015 methodology)"
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor
$P $R/stats.py base=$OUT/base_s42,$OUT/base_s1234 o8=$OUT/o8_s42,$OUT/o8_s1234 2>&1

echo "############### FLAG-ACTUALLY-FIRED CONTROL"
for s in 42 1234; do
  same=0; tot=0
  for f in $OUT/base_s$s/*.pdb; do
    b=$(basename $f); o=$OUT/o8_s$s/$b
    [ -f "$o" ] || continue
    tot=$((tot+1)); cmp -s "$f" "$o" && same=$((same+1))
  done
  echo "FIREDCHECK arm=o8 seed=$s identical_pdbs=$same / $tot"
done
rm -rf $OUT

echo "===== speed re-baseline L=100 (exact vs fast [frame_mulsum still off in the dict] vs fast+O8 via env)"
for preset in exact fast; do
  for r in 1 2; do
    $P -u scripts/bench_one.py --length 100 --no-eager --preset $preset --tag headB_${preset}_r$r 2>&1 | grep -E '^RESULT'
  done
done
SALAD_OPT_FRAMEMULSUM=1 $P -u scripts/bench_one.py --length 100 --no-eager --preset fast --tag headB_fast_o8_r1 2>&1 | grep -E '^RESULT'
SALAD_OPT_FRAMEMULSUM=1 $P -u scripts/bench_one.py --length 100 --no-eager --preset fast --tag headB_fast_o8_r2 2>&1 | grep -E '^RESULT'

echo "GPU_SUITE_EXIT=$rc"
rm -rf $W
