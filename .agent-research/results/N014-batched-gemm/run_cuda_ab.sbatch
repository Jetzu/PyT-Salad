#!/bin/bash -l
#SBATCH --job-name=n014_cudaab
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N014-batched-gemm/logs/cudaab_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N014-batched-gemm
md5sum salad_torch/modules/cond_prologue.py salad_torch/opt.py
export SALAD_OPT_AGGRFUSE=0
# SAME PROCESS, same frozen inputs: 8 identical reruns give the floor, then the
# batched arm. Several seeds and lengths, because a tie flip is a property of
# the geometry, not of the change.
for S in 1 2 3 4 5; do
  for L in 50 100 200; do
    $P -u $R/x_p7_cuda_ab.py --device cuda --length $L --seed $S --reps 8 2>&1 \
      | grep -E "^ARM|^  pos |^  local " || true
  done
done
echo "#### the same, with pair conditioning live ####"
for S in 1 2 3; do
  $P -u $R/x_p7_cuda_ab.py --device cuda --length 100 --seed $S --reps 8 --chains 2 2>&1 \
    | grep -E "^ARM|^  pos |^  local " || true
done
