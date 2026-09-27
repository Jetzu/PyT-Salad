#!/bin/bash -l
#SBATCH --job-name=n026parityint
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH -o /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N026-p7-fusion-rootcause/codex/parity_intended_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
D=$ROOT/.agent-research/results/N026-p7-fusion-rootcause/codex
T=${TMPDIR:-/dev/shm}/n026parityint
mkdir -p "$T"
export TMPDIR=$T
export PYTHONPATH=$D:$ROOT${PYTHONPATH:+:$PYTHONPATH}
export SALAD_OPT_BATCHCOND=1
export SALAD_OPT_DETSEG=0
export SALAD_P7_BARRIER=none
export N026_PATCH=barrier
export N026_REQUIRE_P7_EAGER=1
export N026_EAGER_MARKER=$D/parity_intended_p7_marker.txt
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
cd "$ROOT"
echo "node=$(hostname) CPU=48 four configs x12 threads; P7 eligible asserted; custom barrier none"
configs=(default_vp default_vp_scaled default_ve_scaled multimotif_vp)
pids=()
for cfg in "${configs[@]}"; do
  SALAD_CONFIG=$cfg $PY tests/test_model_parity.py > "$D/parity_intended_$cfg.log" 2>&1 &
  pids+=("$!")
done
rc=0
for i in "${!configs[@]}"; do
  cfg=${configs[$i]}
  if wait "${pids[$i]}"; then st=0; else st=1; rc=1; fi
  echo "$cfg exit=$st"
  grep -E '^coordinate deviation:|^OK$|^FAILED$|^SKIP:' "$D/parity_intended_$cfg.log" | tail -5 || true
  if ! grep -q '^OK$' "$D/parity_intended_$cfg.log"; then rc=1; fi
done
[ "$rc" -eq 0 ]
test -s "$N026_EAGER_MARKER"
echo N026_PARITY_ALL_FOUR_PASS_P7_ON
