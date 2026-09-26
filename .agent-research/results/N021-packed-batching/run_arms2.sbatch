#!/bin/bash -l
#SBATCH --job-name=n021_arms2
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching/logs/arms2_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind-$SLURM_JOB_ID
# A CLEAN DETACHED WORKTREE, not the main checkout: two other agents are editing
# the main tree while this runs and a 40-minute measurement must not read a
# half-written file. Designs go to NODE-LOCAL storage: /hnvme is inode-capped
# and in grace, and 640 PDBs there would be careless.
W=/hnvme/workspace/v181bb10-salad/n021tree
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching
cd $W
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
OUT=${TMPDIR:-/dev/shm}/n021_designs-$SLURM_JOB_ID
N=${SALAD_N:-160}
SEED=${SALAD_SEED:-1234}
mkdir -p $OUT
echo "worktree HEAD: $(git -C $W rev-parse HEAD) (detached)"
md5sum salad_torch/modules/fast.py salad_torch/diffusion/sampling.py scripts/generate.py
nvidia-smi --query-gpu=name --format=csv,noheader
# ONE PROCESS PER ARM. The arms differ ONLY in --batch_size; --group_size is
# chosen so every arm advances 8 designs per replay, and grouping is
# bit-identical (N019) so it cannot be a confound.
run () { # name batch group
  echo "########## ARM $1  batch_size=$2 group_size=$3"
  timeout 4000 $P scripts/generate.py --num_aa 100 --num_designs $N --fast \
     --preset fast --seed $SEED --batch_size $2 --group_size $3 \
     --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz \
     --out_path $OUT/$1 2>&1 | grep -E "designs of|grouped fast|NB batch"
}
run a_b1 1 8
run b_b2 2 4
run c_b4 4 2
run d_b8 8 1
echo "########## counts"; for d in a_b1 b_b2 c_b4 d_b8; do echo "$d: $(ls $OUT/$d/*.pdb 2>/dev/null | wc -l)"; done
echo "########## RAW ss.py SUMMARY"
$P $W/.agent-research/results/N011-fast-preset/ss.py $OUT/a_b1 $OUT/b_b2 $OUT/c_b4 $OUT/d_b8
echo "########## STATS (N015 methodology, n=$N per arm)"
$P $R/stats.py $OUT/a_b1 $OUT/b_b2 $OUT/c_b4 $OUT/d_b8
echo "########## AA COMPOSITION (population-level discrete signal)"
$P $R/x_aacomp.py $OUT/a_b1 $OUT/b_b2 $OUT/c_b4 $OUT/d_b8
echo "### done $(date)"
rm -rf $TORCHINDUCTOR_CACHE_DIR $OUT
