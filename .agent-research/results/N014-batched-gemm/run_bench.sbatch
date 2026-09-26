#!/bin/bash -l
#SBATCH --job-name=n014_bench
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:45:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N014-batched-gemm/logs/bench_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
L=${SALAD_L:?}
md5sum salad_torch/modules/cond_prologue.py salad_torch/modules/blocks.py \
       salad_torch/modules/diffusion.py salad_torch/modules/fast.py salad_torch/opt.py
# alternate the arms so clock drift hits both equally; ONE length per process
for r in 1 2 3; do
                        $P -u scripts/bench_one.py --length $L --no-eager --tag base_r$r || true
  SALAD_OPT_BATCHCOND=1 $P -u scripts/bench_one.py --length $L --no-eager --tag p7_r$r   || true
done
