#!/bin/bash -l
#SBATCH --job-name=n014_gates3
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N014-batched-gemm/logs/gates3_%j.log
set -x
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export OMP_NUM_THREADS=48
R=.agent-research/results/N014-batched-gemm
md5sum salad_torch/modules/cond_prologue.py salad_torch/modules/blocks.py \
       salad_torch/modules/diffusion.py salad_torch/modules/fast.py salad_torch/opt.py

# Re-verification with the CURRENT working tree (fast.py/opt.py gained
# DET_SEGMENT/AGGRESSIVE_FUSION + the "deterministic" preset since gates_903274
# ran; cond_prologue.py/blocks.py/diffusion.py are byte-identical to that run -
# confirmed by md5sum). BOTH arms set SALAD_OPT_BATCHCOND EXPLICITLY: opt.py's
# bare default is currently True, so an unset env var is NOT a valid "P7 off"
# baseline on this tree.
echo "################ GATE 1: parity suite, P7 OFF (SALAD_OPT_BATCHCOND=0) ################"
SALAD_OPT_BATCHCOND=0 bash $R/run_parity.sh
echo "################ GATE 1b: parity suite, P7 ON (SALAD_OPT_BATCHCOND=1) ################"
SALAD_OPT_BATCHCOND=1 bash $R/run_parity.sh
