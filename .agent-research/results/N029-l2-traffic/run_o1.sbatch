#!/bin/bash -l
#SBATCH --job-name=n029_o1
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N029-l2-traffic/logs/o1_%j.log
set -u
module load cuda/12.6.2
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N029-l2-traffic
export SALAD_ROOT=/hnvme/workspace/v181bb10-salad/n029-l2
cd $SALAD_ROOT
echo "### node $(hostname) $(date)"
for ARM in 0 1; do
  for L in 100 200; do
    export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/ind_${SLURM_JOB_ID}_${ARM}_$L
    export TRITON_CACHE_DIR=$TMPDIR/tri_${SLURM_JOB_ID}_${ARM}_$L
    SALAD_OPT_PAIRVEC=$ARM $P -u scripts/bench_one.py --length $L --preset exact \
       --no-eager --iters 40 --tag "pairvec=$ARM"
    rm -rf $TMPDIR/ind_${SLURM_JOB_ID}_${ARM}_$L $TMPDIR/tri_${SLURM_JOB_ID}_${ARM}_$L
  done
done
echo "### kernel census both arms (L=100)"
for ARM in 0 1; do
  export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/ic_${SLURM_JOB_ID}_$ARM
  export TRITON_CACHE_DIR=$TMPDIR/tc_${SLURM_JOB_ID}_$ARM
  echo "--- PAIRVEC=$ARM ---"
  N029_OUT=$R/kernel_times_pairvec$ARM.json SALAD_OPT_PAIRVEC=$ARM SALAD_PRESET=exact \
      $P -u $R/scripts/x_prof.py 2>&1 | head -12
  rm -rf $TMPDIR/ic_${SLURM_JOB_ID}_$ARM $TMPDIR/tc_${SLURM_JOB_ID}_$ARM
done
echo "### done $(date)"
