#!/bin/bash -l
#SBATCH --job-name=n018_qrand
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:55:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N018-fast-levers/logs/qrand_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
W=/hnvme/workspace/v181bb10-salad/n018-worktree
cd $W
OUT=/hnvme/workspace/v181bb10-salad/n018_designs
mkdir -p $OUT
echo "worktree HEAD: $(git -C $W rev-parse HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader

# Same two seeds + counts as N015's a_exact (n015_designs/a_exact seed 42 n=40,
# n015_designs2/a_exact seed 1234 n=120), so this pools to n=160 against the
# SAME baseline stats.py already validated, no re-generation of `exact` needed
# - "exact" preset math is unchanged since N015 (comment-only diff since
# c5ab7a0). --num_steps/--out_steps pinned to the full 500/400 schedule so
# this isolates the neighbourhood lever, not steps_scale.
echo "########## fast_randhalf  seed=42  n=40"
python scripts/generate.py --num_aa 100 --num_designs 40 --fast --preset fast_randhalf \
   --num_steps 500 --out_steps 400 --seed 42 --out_path $OUT/randhalf_s42

echo "########## fast_randhalf  seed=1234  n=120"
python scripts/generate.py --num_aa 100 --num_designs 120 --fast --preset fast_randhalf \
   --num_steps 500 --out_steps 400 --seed 1234 --out_path $OUT/randhalf_s1234

echo "########## RAW ss.py SUMMARY"
python /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N011-fast-preset/ss.py \
   $OUT/randhalf_s42 $OUT/randhalf_s1234
