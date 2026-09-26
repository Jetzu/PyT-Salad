#!/bin/bash -l
#SBATCH --job-name=n019_final
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N019-multidesign-occupancy/logs/final_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N019-multidesign-occupancy
T=/hnvme/workspace/v181bb10-salad/n019final
export SALAD_ROOT=$T
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind-$SLURM_JOB_ID
W=${TMPDIR:-/dev/shm}/n019f-$SLURM_JOB_ID
mkdir -p $W
cd $T
echo "### node $(hostname) $(date)  FINAL VERIFICATION of the commit candidate"
md5sum salad_torch/modules/fast.py salad_torch/diffusion/sampling.py scripts/generate.py

echo "===== 1. the standard parity suite (the single-design path must be untouched)"
rc=0
for C in default_vp default_vp_scaled default_ve_scaled multimotif_vp; do
  out=$(SALAD_CONFIG=$C python tests/test_model_parity.py 2>&1); st=$?
  echo "$out" | grep -E "^coordinate deviation|^OK|^FAIL|FAILED|SKIP" | sed "s/^/[$C] /"
  echo "[$C] exit=$st"; [ $st -ne 0 ] && rc=1
done
for TT in test_geometry_parity test_stack_parity test_encoder_parity test_sampling_parity; do
  out=$(python tests/$TT.py 2>&1); st=$?
  echo "$out" | tail -3 | sed "s/^/[$TT] /"
  echo "[$TT] exit=$st"; [ $st -ne 0 ] && rc=1
done
echo "SUITE_EXIT=$rc"

echo "===== 2. the identity gate, DETSEG, full schedule, 8 designs, BOTH presets"
export SALAD_OPT_DETSEG=1
for PRE in exact fast; do
  for G in 1 8; do
    timeout 2500 python -u $R/x_identity.py --preset $PRE --group_size $G \
        --num_designs 8 --num_aa 100 --out $W/${PRE}_g$G 2>&1 | tail -3
  done
  python -u $R/cmp_identity.py $W/${PRE}_g1 $W/${PRE}_g8 "FINAL_DETSEG_${PRE}_g1_vs_g8" | tail -2
done
unset SALAD_OPT_DETSEG

echo "===== 3. the CLI surface: auto, a partial trailing group, and the refusals"
timeout 2500 python -u scripts/generate.py --fast --preset fast --num_aa 100 \
  --num_designs 10 --group_size 4 --seed 7 --out_path $W/partial \
  --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz 2>&1 \
  | grep -E "group of|designs of|grouped fast path"
echo "  -> PDBs written: $(ls $W/partial/*.pdb 2>/dev/null | wc -l) (expect 10)"
for BAD in "--sym c4" "--renoise" "--sequence_only" "--sample_final"; do
  python scripts/generate.py --fast --num_aa 100 --num_designs 1 --group_size 2 \
    $BAD --out_path $W/bad 2>&1 | tail -1 | sed "s/^/  refuse[$BAD]: /"
done
python scripts/generate.py --num_aa 100 --num_designs 1 --group_size 2 \
  --out_path $W/bad 2>&1 | tail -1 | sed "s/^/  refuse[no --fast]: /"
echo "### done $(date)"
rm -rf $W $TORCHINDUCTOR_CACHE_DIR
