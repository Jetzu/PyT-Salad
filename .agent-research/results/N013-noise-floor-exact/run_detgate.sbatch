#!/bin/bash -l
#SBATCH --job-name=n013_dg
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N013-noise-floor-exact/logs/detgate_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N013-noise-floor-exact
export SALAD_OPT_DETSEG=1
# With the deterministic segment reduction the GPU forward is bit-reproducible
# (job 903261: 1 distinct pattern in 40), so a candidate can be gated ON THE
# DEVICE by bit-equality instead of by a noise-floor argument.
G="--length 100 --reps 4 --time"
$P -u $R/x_floor.py $G --input synth --tag dg_base  --save $R/dg_synth_base.pt || true
SALAD_IND=aggressive_fusion=1 $P -u $R/x_floor.py $G --input synth --tag dg_aggr --save $R/dg_synth_aggr.pt || true
SALAD_OPT_BATCHCOND=1 $P -u $R/x_floor.py $G --input synth --tag dg_p7 --save $R/dg_synth_p7.pt || true
# and on a REAL mid-trajectory state with self-conditioning LIVE (raw_t 0.90),
# which is the only regime where P7 moves anything at all
$P -u $R/x_floor.py $G --input real --upto 50 --tag dg_rbase \
   --state-save $R/state_t90.pt --save $R/dg_real_base.pt || true
SALAD_IND=aggressive_fusion=1 $P -u $R/x_floor.py $G --state-load $R/state_t90.pt \
   --tag dg_raggr --save $R/dg_real_aggr.pt || true
SALAD_OPT_BATCHCOND=1 $P -u $R/x_floor.py $G --state-load $R/state_t90.pt \
   --tag dg_rp7 --save $R/dg_real_p7.pt || true
# the same real state WITHOUT determinism, to size the floor there
SALAD_OPT_DETSEG=0 $P -u $R/x_floor.py --length 100 --reps 30 --state-load $R/state_t90.pt \
   --tag nd_t90 --save $R/nd_real_t90.pt || true
SALAD_OPT_DETSEG=0 SALAD_OPT_BATCHCOND=1 $P -u $R/x_floor.py --length 100 --reps 30 \
   --state-load $R/state_t90.pt --tag nd_t90_p7 --save $R/nd_real_t90_p7.pt || true
