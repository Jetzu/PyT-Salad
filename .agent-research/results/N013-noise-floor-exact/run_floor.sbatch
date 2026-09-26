#!/bin/bash -l
#SBATCH --job-name=n013_floor
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N013-noise-floor-exact/logs/floor_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N013-noise-floor-exact
$P -u $R/x_detprobe.py --length 100 --feat 512 || true
for L in 50 100 200; do
  $P -u $R/x_floor.py --length $L --reps 40 --input synth --tag synth \
     --save $R/floor_synth_L$L.pt || true
done
for L in 50 100 200; do
  $P -u $R/x_floor.py --length $L --reps 40 --input real --upto 250 --tag real \
     --save $R/floor_real_L$L.pt || true
done
# a late-trajectory state too: the argsort ties are a property of the geometry
$P -u $R/x_floor.py --length 100 --reps 40 --input real --upto 399 --tag real399 \
   --save $R/floor_real399_L100.pt || true
