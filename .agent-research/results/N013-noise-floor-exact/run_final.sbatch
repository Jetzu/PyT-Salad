#!/bin/bash -l
#SBATCH --job-name=n013_fin
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N013-noise-floor-exact/logs/final_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N013-noise-floor-exact
$P -u $R/x_graph.py 100 || true
# the shipped `exact` vs what N009 shipped, one process per length
for L in 50 100 200 400; do
  SALAD_OPT_BATCHCOND=0 SALAD_OPT_AGGRFUSE=0 \
    $P -u scripts/bench_one.py --length $L --tag n009 --no-eager || true
  $P -u scripts/bench_one.py --length $L --tag n013 --no-eager || true
done
$P -u scripts/bench_one.py --length 100 --preset deterministic --tag det --no-eager || true
# end to end
$P -u scripts/generate.py --num_aa 100 --num_designs 3 --fast --preset exact \
   --out_path outputs/n013_exact --seed 42 2>&1 | tail -6 || true
$P -u scripts/generate.py --num_aa 100 --num_designs 3 --fast --preset deterministic \
   --out_path outputs/n013_det1 --seed 42 2>&1 | tail -6 || true
$P -u scripts/generate.py --num_aa 100 --num_designs 3 --fast --preset deterministic \
   --out_path outputs/n013_det2 --seed 42 2>&1 | tail -6 || true
for i in 0 1 2; do
  cmp outputs/n013_det1/design_$i.pdb outputs/n013_det2/design_$i.pdb \
    && echo "SEED-PIN design_$i: IDENTICAL" || echo "SEED-PIN design_$i: DIFFERS"
done
