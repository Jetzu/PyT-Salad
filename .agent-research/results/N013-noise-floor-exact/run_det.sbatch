#!/bin/bash -l
#SBATCH --job-name=n013_det
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N013-noise-floor-exact/logs/det_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N013-noise-floor-exact
F="--length 100 --reps 40 --input synth --time"
                        $P -u $R/x_floor.py $F --tag base2  --save $R/det_base_L100.pt   || true
SALAD_OPT_DETSEG=1      $P -u $R/x_floor.py $F --tag detseg --save $R/det_seg_L100.pt    || true
SALAD_TORCH_DET=1       $P -u $R/x_floor.py $F --tag torchdet --save $R/det_torch_L100.pt || true
# length sweep for the two candidates that may ship
for L in 50 200; do
  T="--length $L --reps 4 --input synth --time"
                              $P -u $R/x_floor.py $T --tag base   || true
  SALAD_IND=aggressive_fusion=1 $P -u $R/x_floor.py $T --tag aggr || true
  SALAD_OPT_BATCHCOND=1       $P -u $R/x_floor.py $T --tag p7     || true
done
