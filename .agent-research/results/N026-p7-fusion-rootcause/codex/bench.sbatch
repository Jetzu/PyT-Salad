#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH -o /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N026-p7-fusion-rootcause/codex/bench_%x_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
D=$ROOT/.agent-research/results/N026-p7-fusion-rootcause/codex
T=${TMPDIR:-/dev/shm}/n026bench
mkdir -p "$T"
export TORCHINDUCTOR_CACHE_DIR=$T/inductor
export TRITON_CACHE_DIR=$T/triton
export PYTHONPATH=$D:$ROOT${PYTHONPATH:+:$PYTHONPATH}
export SALAD_OPT_DETSEG=1
case "$N026_BENCH_ARM" in
  off) export SALAD_OPT_BATCHCOND=0; unset N026_PATCH;;
  broken) export SALAD_OPT_BATCHCOND=1; unset N026_PATCH;;
  barrier) export SALAD_OPT_BATCHCOND=1; export N026_PATCH=barrier;;
  *) exit 2;;
esac
cd "$ROOT"
echo "node=$(hostname) arm=$N026_BENCH_ARM tmp=$T"
$PY "$D/bench_p7.py" --length 100 --preset fast --iters 30 --no-eager --tag "$N026_BENCH_ARM"
echo N026_BENCH_DONE
