#!/bin/bash -l
#SBATCH --job-name=n021_ctl
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching/logs/ctl_%j.log
set -u
# CONTROL for the n=160 quality gate. The a_b1 reference arm came out at
# helix 0.89 / 26-of-160 diverged, against N015's recorded fast-preset numbers
# of helix 0.58 / 0-of-160 (measured at commit c5ab7a0). Before ANY of that is
# attributed to packing, establish where the difference lives:
#   L1  legacy --fast on UNMODIFIED HEAD (fd9ba32)          <- is HEAD itself changed?
#   L2  --group_size 8 on UNMODIFIED HEAD                   <- is it N019's grouping?
#   L3  legacy --fast on the N021 worktree                  <- is it MY change?
# All three: L=100, preset fast, seed 42, n=160, same params file.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind-$SLURM_JOB_ID
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching
BASE=/hnvme/workspace/v181bb10-salad/n021base
MINE=/hnvme/workspace/v181bb10-salad/n021tree
OUT=${TMPDIR:-/dev/shm}/n021ctl-$SLURM_JOB_ID
PAR=/hnvme/workspace/v181bb10-salad/default_vp-200k.npz
mkdir -p $OUT
echo "### node $(hostname) $(date)"
echo "base HEAD: $(git -C $BASE rev-parse HEAD)   mine HEAD: $(git -C $MINE rev-parse HEAD)"
md5sum $PAR
echo "########## L1 legacy --fast, UNMODIFIED HEAD"
cd $BASE && timeout 4000 $P scripts/generate.py --num_aa 100 --num_designs 160 \
  --fast --preset fast --seed 42 --params $PAR --out_path $OUT/L1 2>&1 | grep -E "designs of|schedule scaled"
echo "########## L2 --group_size 8, UNMODIFIED HEAD"
cd $BASE && timeout 4000 $P scripts/generate.py --num_aa 100 --num_designs 160 \
  --fast --preset fast --seed 42 --group_size 8 --params $PAR --out_path $OUT/L2 2>&1 | grep -E "designs of|schedule scaled"
echo "########## L3 legacy --fast, N021 WORKTREE"
cd $MINE && timeout 4000 $P scripts/generate.py --num_aa 100 --num_designs 160 \
  --fast --preset fast --seed 42 --params $PAR --out_path $OUT/L3 2>&1 | grep -E "designs of|schedule scaled"
echo "########## STATS: L1 is the reference"
$P $R/stats.py $OUT/L1 $OUT/L2 $OUT/L3 2>&1 | sed -n '1,40p'
echo "### done $(date)"
rm -rf $TORCHINDUCTOR_CACHE_DIR $OUT
