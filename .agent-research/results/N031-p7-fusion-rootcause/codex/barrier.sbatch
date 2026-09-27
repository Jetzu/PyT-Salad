#!/bin/bash -l
#SBATCH --job-name=n026barrier
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH -o /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N026-p7-fusion-rootcause/codex/barrier_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
D=$ROOT/.agent-research/results/N026-p7-fusion-rootcause/codex
T=${TMPDIR:-/dev/shm}/n026barrier
mkdir -p "$T"
export TORCHINDUCTOR_CACHE_DIR=$T/inductor
export TRITON_CACHE_DIR=$T/triton
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
cd "$ROOT"
echo "node=$(hostname) tmp=$T"
$PY "$D/bisect_forward.py" --root "$ROOT" --state "$T/state.pt" --out "$T/unused.npz" --p7 off --dump > "$D/dump_barrier.log" 2>&1
$PY "$D/bisect_forward.py" --root "$ROOT" --state "$T/state.pt" --out "$T/eager.npz" --p7 on > "$D/eager_barrier.log" 2>&1
$PY "$D/barrier_forward.py" --root "$ROOT" --state "$T/state.pt" --out "$T/barrier.npz" --p7 on --compiled > "$D/barrier_forward.log" 2>&1
$PY - "$T/eager.npz" "$T/barrier.npz" <<'PY'
import sys,numpy as np
a,b=map(np.load,sys.argv[1:]);
for k in ('pos','local','trajectory'): print(k,'max_abs',np.abs(a[k]-b[k]).max())
PY
echo N026_BARRIER_DONE
