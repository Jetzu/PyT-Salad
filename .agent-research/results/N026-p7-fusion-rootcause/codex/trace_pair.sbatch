#!/bin/bash -l
#SBATCH --job-name=n026pair
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH -o /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N026-p7-fusion-rootcause/codex/pair_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
D=$ROOT/.agent-research/results/N026-p7-fusion-rootcause/codex
T=${TMPDIR:-/dev/shm}/n026pair
mkdir -p "$T"
export TORCHINDUCTOR_CACHE_DIR=$T/inductor
export TRITON_CACHE_DIR=$T/triton
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
cd "$ROOT"
echo "node=$(hostname) tmp=$T"
$PY "$D/bisect_forward.py" --root "$ROOT" --state "$T/state.pt" --out "$T/unused.npz" --p7 off --dump > "$D/dump_pair.log" 2>&1
$PY "$D/bisect_forward.py" --root "$ROOT" --state "$T/state.pt" --out "$T/eager.npz" --p7 on > "$D/eager_pair.log" 2>&1
for lim in 14 15; do
  export TORCH_COMPILE_DEBUG=1
  export N026_TRACE_DIR=$T/trace_lim$lim
  $PY "$D/bisect_forward.py" --root "$ROOT" --state "$T/state.pt" --out "$T/lim$lim.npz" --p7 on --compiled --fusion-limit "$lim" > "$D/trace_lim$lim.log" 2>&1
  tar -C "$T" -czf "$D/trace_lim$lim.tar.gz" "trace_lim$lim"
  $PY - "$T/eager.npz" "$T/lim$lim.npz" "$lim" <<'PY'
import sys,numpy as np
a,b=map(np.load,sys.argv[1:3]);print('limit',sys.argv[3],'pos_abs',np.abs(a['pos']-b['pos']).max())
PY
  unset TORCH_COMPILE_DEBUG N026_TRACE_DIR
done
$PY "$D/bisect_forward.py" --root "$ROOT" --state "$T/state.pt" --out "$T/veto.npz" --p7 on --compiled --veto-pair op195 op207 > "$D/veto.log" 2>&1
$PY - "$T/eager.npz" "$T/veto.npz" <<'PY'
import sys,numpy as np
a,b=map(np.load,sys.argv[1:]);print('veto_pair pos_abs',np.abs(a['pos']-b['pos']).max())
PY
$PY "$D/minimal_repro.py" > "$D/minimal_synthetic.log" 2>&1
$PY "$D/minimal_repro.py" --nofuse > "$D/minimal_synthetic_nofuse.log" 2>&1
echo N026_PAIR_DONE
