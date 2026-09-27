#!/bin/bash -l
#SBATCH --job-name=n025_path
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /hnvme/workspace/v181bb10-salad/n025_x/job4_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TMPWORK=/tmp/n025_$SLURM_JOB_ID
mkdir -p $TMPWORK/ind
export TORCHINDUCTOR_CACHE_DIR=$TMPWORK/ind
export TRITON_CACHE_DIR=$TMPWORK/triton
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
X=/hnvme/workspace/v181bb10-salad/n025_x
H=/hnvme/workspace/v181bb10-salad/n025
G=/hnvme/workspace/v181bb10-salad/n025_good
OUT=$TMPWORK/designs
N=32; SEED=42
mkdir -p $OUT
echo "head=$(git -C $H rev-parse --short HEAD) good=$(git -C $G rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader
# every arm on the SAME pinned schedule so only the execution path differs
COMMON="--num_aa 100 --num_designs $N --num_steps 250 --out_steps 200 --seed $SEED"
run () { echo "########## $1"; shift; "$@" 2>&1 | tail -3; }
run "A: HEAD --fast --preset fast (COMPILED, TF32) - the broken config" \
    $P $X/run_gen.py $H $COMMON --fast --preset fast --out_path $OUT/a_head_fast
run "B: HEAD EAGER (no --fast), fp32" \
    $P $X/run_gen.py $H $COMMON --out_path $OUT/b_head_eager
run "C: HEAD --fast --preset exact (COMPILED, fp32)" \
    $P $X/run_gen.py $H $COMMON --fast --preset exact --out_path $OUT/c_head_exact
run "D: HEAD --fast --preset fast, N017/O1 CSE VETOED" \
    env N025_NOCSE=1 $P $X/run_gen.py $H $COMMON --fast --preset fast --out_path $OUT/d_head_nocse
run "E: GOOD --fast --preset fast (COMPILED, TF32)" \
    $P $X/run_gen.py $G $COMMON --fast --preset fast --out_path $OUT/e_good_fast
run "F: GOOD EAGER, fp32" \
    $P $X/run_gen.py $G $COMMON --out_path $OUT/f_good_eager
run "G: GOOD --fast --preset exact (COMPILED, fp32)" \
    $P $X/run_gen.py $G $COMMON --fast --preset exact --out_path $OUT/g_good_exact
echo "########## METRICS"
$P $X/metrics.py $OUT/* | tee $X/traj/quality_arms.txt
rm -rf $TMPWORK
echo "ALL ARMS DONE"
