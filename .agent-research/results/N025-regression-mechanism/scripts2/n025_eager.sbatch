#!/bin/bash -l
#SBATCH --job-name=n25eager
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /hnvme/workspace/v181bb10-salad/n025_eager_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
T=${TMPDIR:-/dev/shm}
export TORCHINDUCTOR_CACHE_DIR=$T/ind
TOOLS=/hnvme/workspace/v181bb10-salad/n025_tools
PAR=/hnvme/workspace/v181bb10-salad/default_vp-200k.npz
OUT=$T/n25e
N=64; SEED=42
mkdir -p $OUT
echo "TMPDIR=$T  node=$(hostname)  N=$N SEED=$SEED"
nvidia-smi --query-gpu=name --format=csv,noheader

arm () {   # label, worktree, extra-generate-args...
  L=$1; W=$2; shift 2
  echo "##### ARM $L  worktree=$(git -C $W rev-parse --short HEAD)  extra='$*'  NOCSE=${SALAD_NOCSE:-unset}"
  rm -rf $OUT/$L; mkdir -p $OUT/$L
  cd $W
  SECONDS=0
  $P $TOOLS/wrap.py scripts/generate.py \
     --config default_vp --params $PAR \
     --num_aa 100 --num_designs $N --seed $SEED --out_path $OUT/$L "$@" 2>&1 | tail -8
  echo "  [wall ${SECONDS}s]"
  $P $TOOLS/metrics.py $OUT/$L $L
  rm -rf $OUT/$L
}

# --- item 1: EAGER (no --fast) at HEAD and at the known-good commit ---------
unset SALAD_NOCSE || true
arm E1_head_EAGER      /hnvme/workspace/v181bb10-salad/n025
arm E2_c5ab7a0_EAGER   /hnvme/workspace/v181bb10-salad/n025_good
# --- anchors on the same seed set -----------------------------------------
arm F3_head_fast       /hnvme/workspace/v181bb10-salad/n025      --fast --preset fast
export SALAD_NOCSE=1
arm F4_head_fast_NOCSE /hnvme/workspace/v181bb10-salad/n025      --fast --preset fast
unset SALAD_NOCSE
# --- eager with P7 vetoed, in case eager is ALSO broken --------------------
export SALAD_OPT_BATCHCOND=0
arm E5_head_EAGER_noP7 /hnvme/workspace/v181bb10-salad/n025
unset SALAD_OPT_BATCHCOND
echo N025_EAGER_DONE
