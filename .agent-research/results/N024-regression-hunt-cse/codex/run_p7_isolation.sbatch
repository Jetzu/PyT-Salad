#!/bin/bash -l
#SBATCH --job-name=n024-p7
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N024-regression-hunt-cse/codex/p7_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_OPT_DETSEG=1
export SALAD_OPT_BATCHCOND=0
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/n024_inductor_p7off_${SLURM_JOB_ID}"
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
R=$ROOT/.agent-research/results/N024-regression-hunt-cse/codex
cd "$ROOT"
echo "node=$(hostname) commit=$(git rev-parse HEAD) date=$(date) P7_OFF"
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u "$R/run_cse_isolation.py" on \
  --fast --preset fast --num_aa 100 --num_designs 32 --seed 42 \
  --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz --out_path "$R/pdb_p7off"
