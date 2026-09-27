#!/bin/bash -l
#SBATCH --job-name=n023_ab
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex/ab_%j.log
set -u
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N023-regression-hunt/codex
export TMPDIR=$PWD/$R/tmp
export TORCHINDUCTOR_CACHE_DIR=$PWD/$R/torch_cache
export TRITON_CACHE_DIR=$PWD/$R/triton_cache
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
for detseg in 0 1; do
export SALAD_OPT_DETSEG=$detseg
for spec in '1 1' '0 1' '1 0' '0 0'; do
    read -r bc af <<< "$spec"
    export SALAD_OPT_BATCHCOND=$bc SALAD_OPT_AGGRFUSE=$af
    echo "### ARM BATCHCOND=$bc AGGRFUSE=$af DETSEG=$SALAD_OPT_DETSEG"
    "$P" tests/test_design_quality.py --n 8 --seed 42 --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz --out_path "$R/out_d${detseg}_b${bc}_a${af}"
    echo "### EXIT=$?"
done
done
