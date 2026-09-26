#!/bin/bash -l
#SBATCH --job-name=n019_e2e
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N019-multidesign-occupancy/logs/e2e_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind-$SLURM_JOB_ID
T=/hnvme/workspace/v181bb10-salad/n019tree
W=${TMPDIR:-/dev/shm}/n019e-$SLURM_JOB_ID
PRE=${SALAD_PRESET:?}
L=${SALAD_L:?}
ND=${SALAD_ND:-8}
BEST=${SALAD_BEST:-8}
mkdir -p $W
cd $T
echo "### node $(hostname) $(date) preset=$PRE L=$L nd=$ND best=$BEST"
# The REAL entry point, timed end to end. ONE length per process throughout.
echo "===== A. legacy --fast (today's production path), $ND designs"
timeout 3300 python -u scripts/generate.py --fast --preset $PRE --num_aa $L \
  --num_designs $ND --seed 42 --out_path $W/legacy \
  --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz \
  2>&1 | grep -E "^ *design|designs of|fast path|grouped|preset " 
echo "===== B. --group_size 1 (grouped machinery, group of one)"
timeout 3300 python -u scripts/generate.py --fast --preset $PRE --num_aa $L \
  --num_designs $ND --seed 42 --out_path $W/g1 --group_size 1 \
  --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz \
  2>&1 | grep -E "^ *design|designs of|group of|grouped|preset |group_size"
echo "===== C. --group_size $BEST"
timeout 3300 python -u scripts/generate.py --fast --preset $PRE --num_aa $L \
  --num_designs $ND --seed 42 --out_path $W/gN --group_size $BEST \
  --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz \
  2>&1 | grep -E "^ *design|designs of|group of|grouped|preset |group_size"
echo "===== D. --group_size auto"
timeout 3300 python -u scripts/generate.py --fast --preset $PRE --num_aa $L \
  --num_designs $ND --seed 42 --out_path $W/gauto --group_size auto \
  --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz \
  2>&1 | grep -E "^ *design|designs of|group of|grouped|preset |group_size"
echo "### done $(date)"
rm -rf $W $TORCHINDUCTOR_CACHE_DIR
