#!/bin/bash -l
#SBATCH --job-name=n021_seedm
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching/logs/seedm_%j.log
set -u
# THE CONFOUND CONTROL for the n=160 arms.
#
# `design_generator(seed, d)` keys a design's RNG stream on its GLOBAL INDEX,
# and the grouped path keys a PACK on its FIRST design. So arm a_b1 (B=1) used
# design indices 0..159, arm b_b2 used 0,2,4,..., c_b4 used 0,4,8,... and d_b8
# used 0,8,16,.... **The four arms therefore sampled four DIFFERENT SETS OF
# SEEDS**, and the monotone helix/divergence trend across them could be a
# property of the seed sets rather than of packing.
#
# This job removes the confound: generate 1280 designs at batch_size=1 (so
# design d really does use stream d), then subsample strides 1, 2, 4 and 8 -
# giving four UNPACKED arms on exactly the seed sets the four packed arms used.
# If the trend reappears here, it is the seeds. If these four are flat, the
# trend in the arms is packing.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind-$SLURM_JOB_ID
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching
W=/hnvme/workspace/v181bb10-salad/n021tree
OUT=${TMPDIR:-/dev/shm}/n021sm-$SLURM_JOB_ID
PAR=/hnvme/workspace/v181bb10-salad/default_vp-200k.npz
mkdir -p $OUT/all
cd $W
echo "### node $(hostname) $(date)"
timeout 5000 $P scripts/generate.py --num_aa 100 --num_designs 1280 --fast \
   --preset fast --seed 42 --batch_size 1 --group_size 8 --params $PAR \
   --out_path $OUT/all 2>&1 | grep -E "designs of|schedule scaled"
echo "generated: $(ls $OUT/all/*.pdb | wc -l)"
for S in 1 2 4 8; do
  mkdir -p $OUT/s$S
  $P - "$OUT" "$S" <<'PY'
import os, shutil, sys
out, s = sys.argv[1], int(sys.argv[2])
for k in range(160):
    src = os.path.join(out, "all", f"design_{k*s}.pdb")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(out, f"s{s}", f"design_{k}.pdb"))
PY
  echo "s$S: $(ls $OUT/s$S/*.pdb 2>/dev/null | wc -l) designs (indices 0,$S,$((2*S)),...)"
done
echo "########## STATS: four UNPACKED arms on the packed arms' seed sets"
$P $R/stats.py $OUT/s1 $OUT/s2 $OUT/s4 $OUT/s8 2>&1 | sed -n '1,26p'
echo "### done $(date)"
rm -rf $TORCHINDUCTOR_CACHE_DIR $OUT
