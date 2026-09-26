#!/bin/bash -l
#SBATCH --job-name=n014_k64
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N014-batched-gemm/logs/k64_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N014-batched-gemm
md5sum salad_torch/modules/cond_prologue.py salad_torch/modules/diffusion.py

echo "######## K=64 (2 chains -> pair conditioning live) vs K=48, L=100 ########"
for r in 1 2; do
                        $P -u $R/bench_p7.py --length 100 --no-eager --chains 2 --tag base64_r$r || true
  SALAD_OPT_BATCHCOND=1 $P -u $R/bench_p7.py --length 100 --no-eager --chains 2 --tag p764_r$r   || true
done
echo "######## kernel census: where the launches went ########"
                        $P -u .agent-research/results/N009-speed-proposals/x_kernels.py --length 100 || true
SALAD_OPT_BATCHCOND=1   $P -u .agent-research/results/N009-speed-proposals/x_kernels.py --length 100 || true
