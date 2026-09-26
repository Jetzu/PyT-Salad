#!/bin/bash -l
#SBATCH --job-name=n018_sp400
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N018-fast-levers/logs/speed400_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
D=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N018-fast-levers
W=/hnvme/workspace/v181bb10-salad/n018-worktree
echo "worktree HEAD: $(git -C $W rev-parse HEAD)  (detached from a8533a7 + N018 diff)"
nvidia-smi --query-gpu=name --format=csv,noheader
# Same-shape configs (K=64) share nothing across processes, but exact+fast are
# run in this job too for a same-day/same-node baseline alongside the two new
# levers - each preset still gets its own process (fresh compile), only the
# sbatch allocation is shared. Fixed schedule is irrelevant here: bench_speed
# times fast()/fast(need_aa=False) directly, never running the sampler loop.
for PRESET in exact fast fast_halfnbr fast_randhalf; do
  echo "########## $PRESET  L=400"
  python -u $D/bench_speed.py --preset $PRESET --length 400 --iters 30 --reps 3
done
