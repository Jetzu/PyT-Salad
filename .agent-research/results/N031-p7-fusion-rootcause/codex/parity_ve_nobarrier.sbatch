#!/bin/bash -l
#SBATCH --job-name=n026venobar
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH -o /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N026-p7-fusion-rootcause/codex/ve_nobarrier_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
export SALAD_CONFIG=default_ve_scaled
export SALAD_OPT_BATCHCOND=1
export SALAD_OPT_DETSEG=0
export SALAD_P7_BARRIER=none
export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48
unset N026_PATCH
cd "$ROOT"
echo "node=$(hostname) BATCHCOND=1 DETSEG=0 P7_BARRIER=none cpu=48"
$PY tests/test_model_parity.py
echo N026_VE_NOBARRIER_PASS
