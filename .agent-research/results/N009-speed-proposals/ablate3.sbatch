#!/bin/bash -l
#SBATCH --job-name=salad_abl3
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:55:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N009-speed-proposals/logs/abl3_%j.log
# Kernel-MERGING knobs only: horizontal (combo) and vertical (aggressive)
# fusion co-schedule work that is already independent, so no element's
# arithmetic changes. Anything that swaps a GEMM algorithm is excluded.
set -e
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
run () { tag=$1; shift; env "$@" $P -u scripts/bench_one.py --length 100 --no-eager --tag "$tag"; }
run ship
run noaux   SALAD_OPT_AUX=0
run aggr    SALAD_IND=aggressive_fusion=1
run combo   SALAD_IND=combo_kernels=1
run both    SALAD_IND=aggressive_fusion=1,combo_kernels=1
run bigfuse SALAD_IND=aggressive_fusion=1,max_fusion_size=256,realize_opcount_threshold=120
echo "=== equivalence for whichever knob wins is run separately"
