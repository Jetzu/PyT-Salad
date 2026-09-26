#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=n017_bench
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N017-triton-audit/bench_%j.log
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n017live
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N017-triton-audit
cd $SALAD_TREE
md5sum salad_torch/modules/blocks.py salad_torch/geometry/segment.py
for rep in 1 2; do
  # one process per arm; inductor cache per arm so a cached graph cannot leak
  N017_TAG=base_r$rep   N017_IDXMEAN=0                TORCHINDUCTOR_CACHE_DIR=/tmp/n017c_${SLURM_JOB_ID}_base   python -u $R/x_bench.py || true
  N017_TAG=idxm_r$rep   N017_IDXMEAN=1                TORCHINDUCTOR_CACHE_DIR=/tmp/n017c_${SLURM_JOB_ID}_idxm   python -u $R/x_bench.py || true
  N017_TAG=cse_r$rep    N017_IDXMEAN=0 N017_CSE=1     TORCHINDUCTOR_CACHE_DIR=/tmp/n017c_${SLURM_JOB_ID}_cse    python -u $R/x_bench.py || true
  N017_TAG=frz_r$rep    N017_IDXMEAN=0 N017_FREEZE=1  TORCHINDUCTOR_CACHE_DIR=/tmp/n017c_${SLURM_JOB_ID}_frz    python -u $R/x_bench.py || true
  N017_TAG=both_r$rep   N017_IDXMEAN=1 N017_CSE=1     TORCHINDUCTOR_CACHE_DIR=/tmp/n017c_${SLURM_JOB_ID}_both   python -u $R/x_bench.py || true
done
