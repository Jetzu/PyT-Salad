#!/bin/bash -l
#SBATCH --job-name=n021_cpusem
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching/logs/cpusem_%j.log
set -u
# CPU ON PURPOSE, and it is not a fallback: this job asks whether packing
# changes the ARITHMETIC, and the CUDA forward's own run-to-run noise is
# 1.755e-02 A (N009) - a thousand times the effect being measured. CPU is
# bit-deterministic, so "0.000e+00" here is a real statement. 48 cores = the
# full NUMA fraction; the work is one eager forward per arm, thread-parallel
# inside torch.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export OMP_NUM_THREADS=48
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching
export SALAD_ROOT=/hnvme/workspace/v181bb10-salad/n021tree
echo "### node $(hostname) $(date)"
echo "########## 1. Does a packed design depend on WHO ELSE is in the pack?"
echo "   (if it does, packing leaks; if it does not, the only effect is the offset)"
for L in 32 50 64 100 200 400; do python -u $R/x_packmate_cpu.py $L; done
echo "########## 2. Do the NEIGHBOUR SETS change? (the B1 sentinel's reach)"
for L in 50 100 200 400; do python -u $R/x_neighbourdiff_cpu.py $L; done
echo "########## 3. WHERE does the first difference appear? (module-level bisect)"
for L in 50 100 200; do python -u $R/x_hookdiff_cpu.py $L; done
echo "### done $(date)"
