#!/bin/bash -l
#SBATCH --job-name=n021_smoke
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching/logs/smoke_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching
T=/hnvme/workspace/v181bb10-salad/n021tree
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind-$SLURM_JOB_ID
W=${TMPDIR:-/dev/shm}/n021s-$SLURM_JOB_ID
mkdir -p $W
cd $T
echo "### node $(hostname) $(date)"
echo "===== A. --batch_size 1 --group_size 2 (must equal today's grouped path)"
timeout 1500 python -u scripts/generate.py --fast --preset fast --num_aa 100 \
  --num_designs 4 --seed 42 --out_path $W/b1 --batch_size 1 --group_size 2 \
  --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz 2>&1 | tail -20
echo "===== B. --batch_size 4 --group_size 1"
timeout 1500 python -u scripts/generate.py --fast --preset fast --num_aa 100 \
  --num_designs 4 --seed 42 --out_path $W/b4 --batch_size 4 --group_size 1 \
  --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz 2>&1 | tail -20
echo "===== C. --batch_size 2 --group_size 2, num_designs 5 (tail path)"
timeout 1500 python -u scripts/generate.py --fast --preset fast --num_aa 100 \
  --num_designs 5 --seed 42 --out_path $W/b2 --batch_size 2 --group_size 2 \
  --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz 2>&1 | tail -24
echo "===== files"; for d in b1 b4 b2; do echo "$d: $(ls $W/$d | wc -l) pdbs"; done
echo "### done $(date)"
rm -rf $TORCHINDUCTOR_CACHE_DIR $W
