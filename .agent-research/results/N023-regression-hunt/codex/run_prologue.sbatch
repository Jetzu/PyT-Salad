#!/bin/bash -l
#SBATCH --job-name=n023_prologue
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:15:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex/prologue_%j.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
R=$PWD/.agent-research/results/N023-regression-hunt/codex
export TMPDIR=$R/tmp TORCHINDUCTOR_CACHE_DIR=$R/torch_cache TRITON_CACHE_DIR=$R/triton_cache
export SALAD_OPT_DETSEG=1 SALAD_OPT_AGGRFUSE=0
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u "$R/prologue_compile.py"
