#!/bin/bash -l
#SBATCH --job-name=n023_zero
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=00:10:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex/zero_%j.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
S=.agent-research/results/N023-regression-hunt/codex/x_p7_zero.py
for args in '--length 100' '--length 100 --chains 2 --pair-cond'; do
    echo "### $args"
    $P "$S" --device cpu $args
    echo "### EXIT=$?"
done
