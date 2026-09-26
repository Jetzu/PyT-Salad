#!/bin/bash -l
#SBATCH --job-name=n013_a2
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N013-noise-floor-exact/logs/arms2_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N013-noise-floor-exact
F="--length 100 --reps 40 --input synth --time"
                              $P -u $R/x_floor.py $F --tag base --save $R/a2_base.pt || true
SALAD_IND=aggressive_fusion=1 $P -u $R/x_floor.py $F --tag aggr --save $R/a2_aggr.pt || true
SALAD_OPT_BATCHCOND=1         $P -u $R/x_floor.py $F --tag p7   --save $R/a2_p7.pt   || true
