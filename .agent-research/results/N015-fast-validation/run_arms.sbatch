#!/bin/bash -l
#SBATCH --job-name=n015_arms
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:55:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N015-fast-validation/logs/arms_%j.log
set -u
# Run from a CLEAN DETACHED WORKTREE at c5ab7a0, not the main checkout: another
# agent is editing salad_torch/modules/{blocks,diffusion}.py + opt.py in the
# main tree while this runs, and a 20-minute measurement must not read a
# half-written file. `git worktree add --detach /hnvme/.../n015-worktree HEAD`.
W=/hnvme/workspace/v181bb10-salad/n015-worktree
cd $W
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
OUT=/hnvme/workspace/v181bb10-salad/n015_designs
N=40
SEED=42
mkdir -p $OUT
echo "worktree HEAD: $(git -C $W rev-parse HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader

# Four arms, one process each (torch.set_float32_matmul_precision is
# process-global, so a fresh process per arm is the only way TF32 cannot leak
# across arms). Same --seed for all four; the arms are NOT paired, because a
# 250-step arm draws a different number of randoms per design than a 500-step
# one, which is why the test below is an unpaired Welch t-test.
echo "########## ARM A exact        (fp32, 500/400)"
$P scripts/generate.py --num_aa 100 --num_designs $N --fast --preset exact \
   --seed $SEED --out_path $OUT/a_exact

echo "########## ARM B tf32_only    (TF32, 500/400)"
$P scripts/generate.py --num_aa 100 --num_designs $N --fast --preset fast \
   --num_steps 500 --out_steps 400 --seed $SEED --out_path $OUT/b_tf32

echo "########## ARM C sched_only   (fp32, 250/200)"
$P scripts/generate.py --num_aa 100 --num_designs $N --fast --preset exact \
   --num_steps 250 --out_steps 200 --seed $SEED --out_path $OUT/c_sched

echo "########## ARM D fast         (TF32, 250/200)"
$P scripts/generate.py --num_aa 100 --num_designs $N --fast --preset fast \
   --seed $SEED --out_path $OUT/d_fast

echo "########## RAW ss.py SUMMARY"
$P $W/.agent-research/results/N011-fast-preset/ss.py \
   $OUT/a_exact $OUT/b_tf32 $OUT/c_sched $OUT/d_fast
