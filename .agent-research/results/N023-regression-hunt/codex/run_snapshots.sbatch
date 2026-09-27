#!/bin/bash -l
#SBATCH --job-name=n023_snap
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex/snap_%j.log
set -u
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
export TMPDIR=$R/tmp
export TORCHINDUCTOR_CACHE_DIR=$R/snapshot_cache
export TRITON_CACHE_DIR=$R/snapshot_triton
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
for rev in c5 603; do
    cd "$R/$rev"
    echo "### SNAPSHOT $rev BATCHCOND=default AGGRFUSE=default DETSEG=default"
    "$P" /home/hpc/v181bb/v181bb10/SALAD_torch/tests/test_design_quality.py --n 8 --seed 42 --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz --out_path "$R/out_$rev"
    echo "### EXIT=$?"
done
