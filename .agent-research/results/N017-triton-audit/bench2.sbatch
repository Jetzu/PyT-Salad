#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --job-name=n017_bench2
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N017-triton-audit/bench2_%j.log
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n017live
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N017-triton-audit
cd $SALAD_TREE
md5sum salad_torch/modules/attention.py salad_torch/geometry/segment.py
# D1: does the IPA contraction rewrite buy anything? (it is NOT bit-exact -
# test_sampling_parity fails on it, job 904010 - so it could only ever be a
# `fast`-preset lever, and only if it pays.)
for rep in 1 2; do
  N017_TAG=base2_r$rep  N017_IDXMEAN=0                        TORCHINDUCTOR_CACHE_DIR=/tmp/n017d_${SLURM_JOB_ID}_b python -u $R/x_bench.py || true
  N017_TAG=ipa_r$rep    N017_IDXMEAN=0 N017_IPA=1             TORCHINDUCTOR_CACHE_DIR=/tmp/n017d_${SLURM_JOB_ID}_i python -u $R/x_bench.py || true
  N017_TAG=cseipa_r$rep N017_IDXMEAN=0 N017_IPA=1 N017_CSE=1  TORCHINDUCTOR_CACHE_DIR=/tmp/n017d_${SLURM_JOB_ID}_c python -u $R/x_bench.py || true
done
# D2: is the CSE win an L=100 artefact? one length per process, as always.
for L in 50 200; do
  N017_LEN=$L N017_TAG=base_L$L N017_IDXMEAN=0            TORCHINDUCTOR_CACHE_DIR=/tmp/n017d_${SLURM_JOB_ID}_b$L python -u $R/x_bench.py || true
  N017_LEN=$L N017_TAG=cse_L$L  N017_IDXMEAN=0 N017_CSE=1 TORCHINDUCTOR_CACHE_DIR=/tmp/n017d_${SLURM_JOB_ID}_c$L python -u $R/x_bench.py || true
done
