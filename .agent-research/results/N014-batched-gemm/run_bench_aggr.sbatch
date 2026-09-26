#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N014-batched-gemm/logs/benchaggr_%j.log
set -x
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
L=${SALAD_L:?}
md5sum salad_torch/modules/cond_prologue.py salad_torch/modules/blocks.py \
       salad_torch/modules/diffusion.py salad_torch/modules/fast.py salad_torch/opt.py
# The missing 4th arm of the D1 2x2 (N014 task): AGGRFUSE alone
# (BATCHCOND=0, AGGRFUSE=1). off/p7/ship already measured in bench2_903621-4.
for r in 1 2 3; do
  SALAD_OPT_BATCHCOND=0 SALAD_OPT_AGGRFUSE=1 $P -u scripts/bench_one.py --length $L --no-eager --tag aggr_r$r || true
done
