#!/bin/bash -l
#SBATCH --job-name=n019_gate
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N019-multidesign-occupancy/logs/gate_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N019-multidesign-occupancy
export SALAD_ROOT=/hnvme/workspace/v181bb10-salad/n019tree
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind-$SLURM_JOB_ID
W=${TMPDIR:-/dev/shm}/n019g-$SLURM_JOB_ID
PRE=${SALAD_PRESET:-exact}
ND=${SALAD_ND:-8}
mkdir -p $W
echo "### node $(hostname) $(date) preset=$PRE ndesigns=$ND"

# ---- arm 1: DETSEG=1. The forward is reproducible by construction, so a
#      grouped design must be BIT-IDENTICAL to the same design run alone.
export SALAD_OPT_DETSEG=1
for G in 1 $ND; do
  timeout 3000 python -u $R/x_identity.py --preset $PRE --group_size $G \
      --num_designs $ND --num_aa 100 --out $W/det_g$G 2>&1 | tail -6
done
python -u $R/cmp_identity.py $W/det_g1 $W/det_g$ND "DETSEG_g1_vs_g$ND"
unset SALAD_OPT_DETSEG

# ---- arm 2: the SHIPPING configuration (atomics). Two things measured in the
#      same job: the self-divergence floor (g1 vs g1, identical command,
#      identical seed - the only free variable left is scatter_add ordering)
#      and the grouped difference (g1 vs gN). The gate is that the second is
#      not larger than the first.
for G in 1 1b $ND; do
  timeout 3000 python -u $R/x_identity.py --preset $PRE --group_size ${G%b} \
      --num_designs $ND --num_aa 100 --out $W/at_g$G 2>&1 | tail -6
done
python -u $R/cmp_identity.py $W/at_g1 $W/at_g1b "ATOMICS_FLOOR_g1_vs_g1"
python -u $R/cmp_identity.py $W/at_g1 $W/at_g$ND "ATOMICS_g1_vs_g$ND"
echo "### done $(date)"
rm -rf $W $TORCHINDUCTOR_CACHE_DIR
