#!/bin/bash -l
#SBATCH --job-name=n013_dg2
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N013-noise-floor-exact/logs/detgate2_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N013-noise-floor-exact
export SALAD_OPT_DETSEG=1
export SALAD_OPT_AGGRFUSE=0          # every arm states its own knobs explicitly
G="--length 100 --reps 4 --time"
# With the deterministic segment reduction the GPU forward is bit-reproducible
# (job 903261: 1 distinct pattern in 40), so a candidate can be gated ON THE
# DEVICE by bit-equality instead of by a noise-floor argument.
SALAD_OPT_BATCHCOND=0 $P -u $R/x_floor.py $G --input synth --tag dg2_base \
   --save $R/dg2_synth_base.pt || true
SALAD_OPT_BATCHCOND=1 $P -u $R/x_floor.py $G --input synth --tag dg2_p7 \
   --save $R/dg2_synth_p7.pt || true
# and on a REAL mid-trajectory state with self-conditioning LIVE (raw_t 0.90),
# the only regime where P7 moves anything at all (x_p7_cpu_sweep.py)
SALAD_OPT_BATCHCOND=0 $P -u $R/x_floor.py $G --input real --upto 50 --tag dg2_rbase \
   --state-save $R/state_t90.pt --save $R/dg2_real_base.pt || true
SALAD_OPT_BATCHCOND=1 $P -u $R/x_floor.py $G --state-load $R/state_t90.pt \
   --tag dg2_rp7 --save $R/dg2_real_p7.pt || true
SALAD_OPT_BATCHCOND=0 SALAD_OPT_AGGRFUSE=1 $P -u $R/x_floor.py $G \
   --state-load $R/state_t90.pt --tag dg2_raggr --save $R/dg2_real_aggr.pt || true
# the same real state WITHOUT determinism, to size the floor exactly there
SALAD_OPT_DETSEG=0 SALAD_OPT_BATCHCOND=0 $P -u $R/x_floor.py --length 100 --reps 30 \
   --state-load $R/state_t90.pt --tag nd_t90 --save $R/nd_real_t90.pt || true
