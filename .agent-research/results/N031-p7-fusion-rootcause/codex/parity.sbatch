#!/bin/bash -l
#SBATCH --job-name=n026parity
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH -o /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N026-p7-fusion-rootcause/codex/parity_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
D=$ROOT/.agent-research/results/N026-p7-fusion-rootcause/codex
T=${TMPDIR:-/dev/shm}/n026parity
mkdir -p "$T"
export TMPDIR=$T
export PYTHONPATH=$D:$ROOT${PYTHONPATH:+:$PYTHONPATH}
export SALAD_OPT_BATCHCOND=1
export SALAD_OPT_DETSEG=1
export N026_PATCH=barrier
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
cd "$ROOT"
echo "node=$(hostname) CPU=48 four configs x12 threads"
configs=(default_vp default_vp_scaled default_ve_scaled multimotif_vp)
pids=()
for cfg in "${configs[@]}"; do
  SALAD_CONFIG=$cfg $PY tests/test_model_parity.py > "$D/parity_$cfg.log" 2>&1 &
  pids+=("$!")
done
rc=0
for i in "${!configs[@]}"; do
  cfg=${configs[$i]}
  wait "${pids[$i]}" || rc=1
  echo "$cfg exit=$?"
  grep -E '^config:|^coordinate deviation:|^OK$|^FAILED$|^SKIP:' "$D/parity_$cfg.log" | tail -5 || true
  if ! grep -q '^OK$' "$D/parity_$cfg.log"; then rc=1; fi
done
[ "$rc" -eq 0 ]
echo N026_PARITY_ALL_FOUR_PASS
