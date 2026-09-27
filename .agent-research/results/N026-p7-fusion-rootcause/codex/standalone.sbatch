#!/bin/bash -l
#SBATCH --job-name=n026standalone
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH -o /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N026-p7-fusion-rootcause/codex/standalone_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
D=$ROOT/.agent-research/results/N026-p7-fusion-rootcause/codex
T=${TMPDIR:-/dev/shm}/n026standalone
mkdir -p "$T"
export TORCHINDUCTOR_CACHE_DIR=$T/inductor
export TRITON_CACHE_DIR=$T/triton
cd "$ROOT"
$PY "$D/standalone_kernel.py"
echo N026_STANDALONE_DONE
