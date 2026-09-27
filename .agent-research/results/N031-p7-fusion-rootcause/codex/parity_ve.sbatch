#!/bin/bash -l
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH -o /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N026-p7-fusion-rootcause/codex/ve_%x_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
D=$ROOT/.agent-research/results/N026-p7-fusion-rootcause/codex
export PYTHONPATH=$D:$ROOT${PYTHONPATH:+:$PYTHONPATH}
export SALAD_CONFIG=default_ve_scaled
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
if [ "$N026_VE_ARM" = on ]; then
  export SALAD_OPT_BATCHCOND=1 N026_PATCH=barrier
else
  export SALAD_OPT_BATCHCOND=0
  unset N026_PATCH
fi
cd "$ROOT"
echo "node=$(hostname) arm=$N026_VE_ARM cpu=48"
$PY tests/test_model_parity.py
echo N026_VE_PASS
