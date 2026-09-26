#!/bin/bash -l
#SBATCH --job-name=n014_bench2
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N014-batched-gemm/logs/bench2_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
L=${SALAD_L:?}
md5sum salad_torch/modules/cond_prologue.py salad_torch/modules/blocks.py \
       salad_torch/modules/diffusion.py salad_torch/modules/fast.py salad_torch/opt.py
# EVERY flag set explicitly - the BATCHCOND default changed under an earlier
# run of this benchmark and silently made both arms identical.
# arms: P7 off / P7 on (aggressive_fusion pinned OFF so P7 is isolated),
#       then the shipping combination.
for r in 1 2 3; do
  SALAD_OPT_BATCHCOND=0 SALAD_OPT_AGGRFUSE=0 $P -u scripts/bench_one.py --length $L --no-eager --tag off_r$r  || true
  SALAD_OPT_BATCHCOND=1 SALAD_OPT_AGGRFUSE=0 $P -u scripts/bench_one.py --length $L --no-eager --tag p7_r$r   || true
  SALAD_OPT_BATCHCOND=1 SALAD_OPT_AGGRFUSE=1 $P -u scripts/bench_one.py --length $L --no-eager --tag ship_r$r || true
done
