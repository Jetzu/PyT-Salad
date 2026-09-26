#!/bin/bash -l
#SBATCH --job-name=n014_gates
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=03:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N014-batched-gemm/logs/gates_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
export OMP_NUM_THREADS=48
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N014-batched-gemm
md5sum salad_torch/modules/cond_prologue.py salad_torch/modules/blocks.py \
       salad_torch/modules/diffusion.py salad_torch/modules/fast.py salad_torch/opt.py

echo "################ GATE 1: parity suite, P7 OFF (the exact default) ################"
                      bash $R/run_parity.sh
echo "################ GATE 1b: parity suite, P7 ON ################"
SALAD_OPT_BATCHCOND=1 bash $R/run_parity.sh

echo "################ GATE 2: CPU torch.equal matrix ################"
for C in default_vp default_vp_scaled default_ve_scaled multimotif_vp; do
  $P -u $R/x_p7_equiv.py --device cpu --length 48 --config $C                      || true
  $P -u $R/x_p7_equiv.py --device cpu --length 48 --config $C --chains 3           || true
  $P -u $R/x_p7_equiv.py --device cpu --length 48 --config $C --pair-cond          || true
  $P -u $R/x_p7_equiv.py --device cpu --length 48 --config $C --pair-cond --chains 3 || true
  $P -u $R/x_p7_equiv.py --device cpu --length 48 --config $C --cyclic             || true
done
# a longer one, where the GEMMs are big enough to pick a split-K
$P -u $R/x_p7_equiv.py --device cpu --length 128 --config default_vp || true
$P -u $R/x_p7_equiv.py --device cpu --length 200 --config default_vp || true
