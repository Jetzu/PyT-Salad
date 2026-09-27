#!/bin/bash -l
#SBATCH --job-name=n026bisect
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH -o /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N026-p7-fusion-rootcause/codex/bisect_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
D=$ROOT/.agent-research/results/N026-p7-fusion-rootcause/codex
export N026_TMP=${TMPDIR:-/dev/shm}/n026bisect
mkdir -p "$N026_TMP"
export TORCHINDUCTOR_CACHE_DIR=$N026_TMP/inductor
export TRITON_CACHE_DIR=$N026_TMP/triton
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
cd "$ROOT"
echo "node=$(hostname) tmp=$N026_TMP"
nvidia-smi --query-gpu=name --format=csv,noheader
$PY "$D/run_bisect.py"
echo N026_BISECT_DONE
