#!/bin/bash -l
#SBATCH --job-name=n026quality
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH -o /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N026-p7-fusion-rootcause/codex/quality_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
PY=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
D=$ROOT/.agent-research/results/N026-p7-fusion-rootcause/codex
T=${TMPDIR:-/dev/shm}/n026quality
mkdir -p "$T"
export TMPDIR=$T
export TORCHINDUCTOR_CACHE_DIR=$T/inductor
export TRITON_CACHE_DIR=$T/triton
export PYTHONPATH=$D:$ROOT${PYTHONPATH:+:$PYTHONPATH}
export SALAD_OPT_DETSEG=1
export SALAD_OPT_BATCHCOND=1
export N026_PATCH=barrier
export N026_REQUIRE_P7=1
export N026_MARKER=$D/quality_p7_marker.txt
cd "$ROOT"
echo "node=$(hostname) tmp=$T"
$PY tests/test_design_quality.py --n 160 --preset fast --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz | tee "$D/quality_detail.log"
grep -q '^diverged: 0/160' "$D/quality_detail.log"
test -s "$N026_MARKER"
echo N026_QUALITY_0_OF_160_P7_ON
