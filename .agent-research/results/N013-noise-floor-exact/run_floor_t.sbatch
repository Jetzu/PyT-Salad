#!/bin/bash -l
#SBATCH --job-name=n013_ft
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N013-noise-floor-exact/logs/floort_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N013-noise-floor-exact
# the floor is a function of the GEOMETRY: a Gaussian blob has near-ties in the
# neighbour argsort, a real backbone does not. Sweep the trajectory.
for U in 5 25 50 100 150 300 399; do
  $P -u $R/x_floor.py --length 100 --reps 30 --input real --upto $U --tag t$U || true
done
