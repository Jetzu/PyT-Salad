#!/bin/bash -l
#SBATCH --job-name=n025_path
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /hnvme/workspace/v181bb10-salad/n025_x/job1_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=/hnvme/workspace/v181bb10-salad/n025_cache
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
X=/hnvme/workspace/v181bb10-salad/n025_x
H=/hnvme/workspace/v181bb10-salad/n025
G=/hnvme/workspace/v181bb10-salad/n025_good
OUT=$X/designs
N=32; SEED=42
rm -rf $OUT; mkdir -p $OUT
echo "head=$(git -C $H rev-parse --short HEAD) good=$(git -C $G rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader
COMMON="--num_aa 100 --num_designs $N --num_steps 250 --out_steps 200 --seed $SEED"

echo "########## A: HEAD --fast --preset fast  (COMPILED, TF32, the broken config)"
$P $X/run_gen.py $H $COMMON --fast --preset fast --out_path $OUT/a_fast_compiled 2>&1 | tail -3
echo "########## B: HEAD EAGER (no --fast), fp32"
$P $X/run_gen.py $H $COMMON --out_path $OUT/b_eager 2>&1 | tail -3
echo "########## C: HEAD --fast --preset exact (COMPILED, fp32)"
$P $X/run_gen.py $H $COMMON --fast --preset exact --out_path $OUT/c_exact_compiled 2>&1 | tail -3
echo "########## D: HEAD --fast --preset fast, N017/O1 CSE pass VETOED"
N025_NOCSE=1 $P $X/run_gen.py $H $COMMON --fast --preset fast --out_path $OUT/d_fast_nocse 2>&1 | tail -3
echo "########## E: GOOD c5ab7a0 --fast --preset fast (COMPILED, TF32) - same pinned schedule"
$P $X/run_gen.py $G $COMMON --fast --preset fast --out_path $OUT/e_good_fast 2>&1 | tail -3
echo "########## F: GOOD c5ab7a0 EAGER, fp32"
$P $X/run_gen.py $G $COMMON --out_path $OUT/f_good_eager 2>&1 | tail -3
echo "########## METRICS"
$P $X/metrics.py $OUT/*
echo "ALL ARMS DONE"
