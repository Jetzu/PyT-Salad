#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --job-name=n030_qual
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas/j5_quality_%j.log
# PLACEMENT: 1x H100, 16 cores, <=8 GPU-h (two n=160 arms + compiles). This is
# the ONLY gate that drives the COMPILED path - all four parity tests are eager
# (K021), which is how P7 destroyed 20 % of designs while passing everything.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export PYTHONHASHSEED=0
export TMPDIR=/dev/shm/n030q_$SLURM_JOB_ID
mkdir -p $TMPDIR
cd /hnvme/workspace/v181bb10-salad/n030
md5sum salad_torch/gemm_exact.py salad_torch/modules/fast.py salad_torch/opt.py
# arm 1: the shipped default on this tree (the reference: must be 0/160)
echo "########## QUALITY ARM=baseline"
TORCHINDUCTOR_CACHE_DIR=$TMPDIR/ind_base python -u tests/test_design_quality.py --n 160 --preset fast
echo "QUALITY_EXIT=$? ARM=baseline"
rm -rf $TMPDIR/ind_base
# arm 2: + decompose_mm_pass at min-batch 32
echo "########## QUALITY ARM=decomp32"
SALAD_OPT_DECOMPBMM=32 TORCHINDUCTOR_CACHE_DIR=$TMPDIR/ind_dec python -u tests/test_design_quality.py --n 160 --preset fast
echo "QUALITY_EXIT=$? ARM=decomp32"
rm -rf $TMPDIR
