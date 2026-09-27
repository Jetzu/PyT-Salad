#!/bin/bash -l
#SBATCH --job-name=n026capture
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH -o /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N026-p7-fusion-rootcause/codex/capture_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
D=$ROOT/.agent-research/results/N026-p7-fusion-rootcause/codex
T=${TMPDIR:-/dev/shm}/n026capture
mkdir -p "$T"
export TORCHINDUCTOR_CACHE_DIR=$T/inductor
export TRITON_CACHE_DIR=$T/triton
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
cd "$ROOT"
echo "node=$(hostname) tmp=$T"
$PY "$D/bisect_forward.py" --root "$ROOT" --state "$T/state.pt" --out "$T/unused.npz" --p7 off --dump > "$D/dump_capture.log" 2>&1
$PY "$D/capture_kernel.py" --root "$ROOT" --state "$T/state.pt" --out "$T/compiled.npz" --p7 on --compiled --fusion-limit 15 --capture "$D/real_kernel_io.pt" > "$D/capture_forward.log" 2>&1
$PY "$D/check_kernel_io.py" "$D/real_kernel_io.pt" > "$D/kernel_numerics.txt"
echo N026_CAPTURE_DONE
