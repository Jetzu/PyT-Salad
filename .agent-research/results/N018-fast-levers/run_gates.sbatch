#!/bin/bash -l
#SBATCH --job-name=n018_gates
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:40:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N018-fast-levers/logs/gates_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
# Sanity-check the WORKTREE (opt.NBR_MODE="full" default + the two new preset
# entries) against the same hard gates N015 ran on its own worktree - my edit
# only touches `Diffusion.forward_core`'s neighbour_fn (used by FastDenoiser),
# never the eager `forward()` these tests exercise, and "full" mode reproduces
# the original (16,16,32)/(16,16,16) literals exactly - so this should be a
# byte-for-byte no-op. Checking it, not assuming it.
bash /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N015-fast-validation/run_gates.sh \
     /hnvme/workspace/v181bb10-salad/n018-worktree
