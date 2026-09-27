#!/bin/bash -l
#SBATCH --job-name=n026trace
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH -o /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N026-p7-fusion-rootcause/codex/trace_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
D=$ROOT/.agent-research/results/N026-p7-fusion-rootcause/codex
T=${TMPDIR:-/dev/shm}/n026trace
mkdir -p "$T"
export TORCHINDUCTOR_CACHE_DIR=$T/inductor
export TRITON_CACHE_DIR=$T/triton
export TORCH_COMPILE_DEBUG=1
export TORCHINDUCTOR_TRACE=1
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export TORCH_LOGS=fusion
cd "$ROOT"
echo "node=$(hostname) tmp=$T"
nvidia-smi --query-gpu=name --format=csv,noheader
$PY "$D/trace_forward.py" --root "$ROOT" --state "$T/state.pt" --out "$T/dump.npz" --p7 off --dump
for arm in off on; do
  export N026_TRACE_DIR=$T/trace_$arm
  $PY "$D/trace_forward.py" --root "$ROOT" --state "$T/state.pt" --out "$T/$arm.npz" --p7 "$arm" --compiled > "$D/$arm.forward.log" 2>&1
  echo "$arm forward done"
  find "$T/trace_$arm" -name output_code.py -o -name ir_post_fusion.txt | head
  tar -C "$T" -czf "$D/trace_$arm.tar.gz" "trace_$arm"
done
$PY - "$T/off.npz" "$T/on.npz" <<'PY'
import sys,numpy as np
a,b=map(np.load,sys.argv[1:])
for k in a.files:
 print(k, np.max(np.abs(a[k]-b[k])))
PY
echo N026_TRACE_DONE
