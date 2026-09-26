#!/bin/bash -l
#SBATCH --job-name=salad_abl2
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:55:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N009-speed-proposals/logs/abl2_%j.log
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
run () { tag=$1; shift; env "$@" $P -u scripts/bench_one.py --length 100 --no-eager --tag "$tag"; }
# P0 = the DotDict graph-break fix + one-shot uniform unbind, now unconditional.
# `base` here == `base` in job 903079 except for P0, so the two jobs bracket it.
run base    SALAD_OPT_GATHER=0 SALAD_OPT_HOIST=0 SALAD_OPT_NOPC=0
run base2   SALAD_OPT_GATHER=0 SALAD_OPT_HOIST=0 SALAD_OPT_NOPC=0
run +P3     SALAD_OPT_GATHER=1 SALAD_OPT_HOIST=0 SALAD_OPT_NOPC=0
run +P6     SALAD_OPT_GATHER=1 SALAD_OPT_HOIST=1 SALAD_OPT_NOPC=0
run +P4P5   SALAD_OPT_GATHER=1 SALAD_OPT_HOIST=1 SALAD_OPT_NOPC=1
run all2    SALAD_OPT_GATHER=1 SALAD_OPT_HOIST=1 SALAD_OPT_NOPC=1
run all3    SALAD_OPT_GATHER=1 SALAD_OPT_HOIST=1 SALAD_OPT_NOPC=1
echo "=== other lengths, shipping config"
for L in 50 200 400; do $P -u scripts/bench_one.py --length $L --no-eager --tag len$L; done
echo "=== equivalence, on the GPU, with a noise-floor control"
$P -u .agent-research/results/N009-speed-proposals/x_equiv.py --device cuda
echo "=== kernel census, shipping config"
$P -u .agent-research/results/N009-speed-proposals/x_kernels.py --length 100
echo "=== kernel census, every proposal OFF"
env SALAD_OPT_GATHER=0 SALAD_OPT_HOIST=0 SALAD_OPT_NOPC=0 SALAD_OPT_LEAN=0 \
  $P -u .agent-research/results/N009-speed-proposals/x_kernels.py --length 100
