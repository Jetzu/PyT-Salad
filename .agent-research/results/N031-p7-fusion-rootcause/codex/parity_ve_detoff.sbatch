#!/bin/bash -l
#SBATCH --job-name=n026vedetoff
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH -o /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N026-p7-fusion-rootcause/codex/ve_detoff_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
D=$ROOT/.agent-research/results/N026-p7-fusion-rootcause/codex
export PYTHONPATH=$D:$ROOT${PYTHONPATH:+:$PYTHONPATH}
export SALAD_CONFIG=default_ve_scaled
export SALAD_OPT_BATCHCOND=1
export SALAD_OPT_DETSEG=0
export N026_PATCH=barrier
export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48
cd "$ROOT"
echo "node=$(hostname) BATCHCOND=1 DETSEG=0 cpu=48"
$PY tests/test_model_parity.py
echo N026_VE_DETOFF_PASS
