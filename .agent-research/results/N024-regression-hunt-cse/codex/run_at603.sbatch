#!/bin/bash -l
#SBATCH --job-name=n024-603
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N024-regression-hunt-cse/codex/at603_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_OPT_DETSEG=1
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/n024_inductor_603_${SLURM_JOB_ID}"
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N024-regression-hunt-cse/codex
cd "$R/at603"
echo "snapshot_commit=603051e node=$(hostname) date=$(date)"
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u scripts/generate.py \
  --fast --preset fast --num_aa 100 --num_designs 32 --seed 42 \
  --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz --out_path "$R/pdb_at603"
