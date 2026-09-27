#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --job-name=n030_ship
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas/j8_shipquality_%j.log
# PLACEMENT: 1x H100, 16 cores, <1 GPU-h. Wiring check: confirm the SHIPPED
# `fast` preset default (decompose_small_bmm=32, no env override at all) gives
# the same n=160 result as j5's env-override arm - i.e. resolve_decompose_
# small_bmm(32) from the preset dict actually threads through, not just the
# SALAD_OPT_DECOMPBMM=32 env path j5 tested.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export PYTHONHASHSEED=0
export TMPDIR=/dev/shm/n030s_$SLURM_JOB_ID
mkdir -p $TMPDIR
cd /hnvme/workspace/v181bb10-salad/n030
md5sum salad_torch/gemm_exact.py salad_torch/modules/fast.py salad_torch/opt.py
echo "########## SHIP QUALITY ARM=fast_default (no env override)"
TORCHINDUCTOR_CACHE_DIR=$TMPDIR/ind python -u tests/test_design_quality.py --n 160 --preset fast
echo "QUALITY_EXIT=$? ARM=fast_default"
rm -rf $TMPDIR
