#!/bin/bash -l
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH -o /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N026-p7-fusion-rootcause/codex/clean_%x_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
export SALAD_CONFIG=$N026_CONFIG
export SALAD_OPT_BATCHCOND=1
export SALAD_OPT_DETSEG=0
export SALAD_P7_BARRIER=none
export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48
unset N026_PATCH N026_REQUIRE_P7_EAGER
cd "$ROOT"
echo "node=$(hostname) config=$N026_CONFIG cpu=48 P7 env on custom barrier none"
$PY -c 'from salad_torch import opt; print("BATCH_COND",opt.BATCH_COND)'
$PY tests/test_model_parity.py
echo N026_CLEAN_PARITY_PASS
