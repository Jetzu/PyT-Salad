#!/bin/bash -l
#SBATCH --job-name=salad_abl
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:55:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N009-speed-proposals/logs/abl_%j.log
# One process per rung - compiling several variants in one process degrades
# cudagraphs and flattens everything (N007 appendix).
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
run () { tag=$1; shift; env "$@" $P -u scripts/bench_one.py --length 100 --no-eager --tag "$tag"; }
run base   SALAD_OPT_GATHER=0 SALAD_OPT_HOIST=0 SALAD_OPT_NOPC=0
run base2  SALAD_OPT_GATHER=0 SALAD_OPT_HOIST=0 SALAD_OPT_NOPC=0
run +P3    SALAD_OPT_GATHER=1 SALAD_OPT_HOIST=0 SALAD_OPT_NOPC=0
run +P6    SALAD_OPT_GATHER=1 SALAD_OPT_HOIST=1 SALAD_OPT_NOPC=0
run +P4P5  SALAD_OPT_GATHER=1 SALAD_OPT_HOIST=1 SALAD_OPT_NOPC=1
run all2   SALAD_OPT_GATHER=1 SALAD_OPT_HOIST=1 SALAD_OPT_NOPC=1
run all3   SALAD_OPT_GATHER=1 SALAD_OPT_HOIST=1 SALAD_OPT_NOPC=1
$P -u .agent-research/results/N009-speed-proposals/x_equiv.py --device cuda
