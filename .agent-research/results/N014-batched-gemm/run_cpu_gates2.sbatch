#!/bin/bash -l
#SBATCH --job-name=n014_gates2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N014-batched-gemm/logs/gates2_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export OMP_NUM_THREADS=48
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N014-batched-gemm
md5sum salad_torch/modules/cond_prologue.py salad_torch/modules/diffusion.py
echo "######## GATE 2b: padding mask and multi-BATCH (the tiling offsets both) ########"
$P -u $R/x_p7_equiv2.py --device cpu --length 64 --mask-frac 0.7                  || true
$P -u $R/x_p7_equiv2.py --device cpu --length 64 --batches 2                      || true
$P -u $R/x_p7_equiv2.py --device cpu --length 64 --batches 2 --chains 4           || true
$P -u $R/x_p7_equiv2.py --device cpu --length 64 --batches 2 --chains 4 --mask-frac 0.8 --pair-cond || true
$P -u $R/x_p7_equiv2.py --device cpu --length 64 --config default_vp_scaled --batches 2 --chains 4 --mask-frac 0.8 || true
