#!/bin/bash -l
#SBATCH --job-name=n013_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=01:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N013-noise-floor-exact/logs/cpubits_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export OMP_NUM_THREADS=48
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N013-noise-floor-exact
# 1. is the compiled CPU forward itself deterministic? (the premise of the gate)
SALAD_IND= $P -u $R/x_cpu_bits.py --length 64 --out $R/cpu_base.pt
SALAD_IND= $P -u $R/x_cpu_bits.py --length 64 --out $R/cpu_base2.pt
$P -u $R/cmp_bits.py $R/cpu_base.pt $R/cpu_base2.pt
# 2. aggressive_fusion=1 against it, bit for bit
SALAD_IND=aggressive_fusion=1 $P -u $R/x_cpu_bits.py --length 64 --out $R/cpu_aggr.pt
$P -u $R/cmp_bits.py $R/cpu_base.pt $R/cpu_aggr.pt
# 3. and at a second length, in case the schedule only differs for some shapes
SALAD_IND= $P -u $R/x_cpu_bits.py --length 100 --out $R/cpu_base_L100.pt
SALAD_IND=aggressive_fusion=1 $P -u $R/x_cpu_bits.py --length 100 --out $R/cpu_aggr_L100.pt
$P -u $R/cmp_bits.py $R/cpu_base_L100.pt $R/cpu_aggr_L100.pt
