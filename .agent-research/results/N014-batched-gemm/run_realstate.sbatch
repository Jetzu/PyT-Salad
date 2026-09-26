#!/bin/bash -l
#SBATCH --job-name=n014_real
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N014-batched-gemm/logs/real_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N014-batched-gemm
export SALAD_OPT_AGGRFUSE=0
echo "#### the exact state the 2.72 A came from ####"
$P -u $R/x_p7_realstate.py --state $R/st2_L100.pt --reps 12 || true
echo "#### and the CPU reading of the SAME state - deterministic, so it is the truth ####"
OMP_NUM_THREADS=16 $P -u $R/x_p7_realstate.py --state $R/st2_L100.pt --reps 3 --device cpu || true
