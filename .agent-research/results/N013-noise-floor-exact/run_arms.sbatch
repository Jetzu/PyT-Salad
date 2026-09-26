#!/bin/bash -l
#SBATCH --job-name=n013_arms
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N013-noise-floor-exact/logs/arms_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N013-noise-floor-exact
F="--length 100 --reps 40 --input synth --time"
# every arm on the SAME frozen synthetic input, one process each, same job
                                 $P -u $R/x_floor.py $F --tag base   --save $R/arm_base_L100.pt   || true
SALAD_IND=aggressive_fusion=1    $P -u $R/x_floor.py $F --tag aggr   --save $R/arm_aggr_L100.pt   || true
SALAD_OPT_BATCHCOND=1            $P -u $R/x_floor.py $F --tag p7     --save $R/arm_p7_L100.pt     || true
SALAD_OPT_BATCHCOND=1 SALAD_IND=aggressive_fusion=1 \
                                 $P -u $R/x_floor.py $F --tag p7aggr --save $R/arm_p7aggr_L100.pt || true
SALAD_OPT_AUX=0                  $P -u $R/x_floor.py $F --tag noaux  --save $R/arm_noaux_L100.pt  || true
