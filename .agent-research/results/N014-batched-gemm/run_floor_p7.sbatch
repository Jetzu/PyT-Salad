#!/bin/bash -l
#SBATCH --job-name=n014_floor
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N014-batched-gemm/logs/floor_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N014-batched-gemm
md5sum salad_torch/modules/cond_prologue.py salad_torch/modules/diffusion.py

echo "######## is the bmm itself a NEW noise source? ########"
for K in 48 64; do
  $P -u $R/x_p7_detprobe.py --length 100 --reps 40 --k $K            || true
  $P -u $R/x_p7_detprobe.py --length 100 --reps 40 --k $K --pair-cond || true
done

echo "######## paired floor: base vs P7 on ONE frozen input, 40 reps each ########"
# freeze a mid-trajectory state once; BOTH arms load it, so the only difference
# between the two rep-sets is the code path
$P -u $R/x_floor.py --length 100 --reps 40 --input real --upto 250 \
   --state-save $R/state_real_L100.pt --tag base --save $R/fl_base_L100.pt || true
SALAD_OPT_BATCHCOND=1 $P -u $R/x_floor.py --length 100 --reps 40 --input real \
   --state-load $R/state_real_L100.pt --tag p7 --save $R/fl_p7_L100.pt || true
$P -u $R/x_floor.py --length 100 --reps 40 --input real \
   --state-load $R/state_real_L100.pt --tag base2 --save $R/fl_base2_L100.pt || true
for K in pos local trajectory; do
  echo "==== within/cross, key=$K : base vs p7 ===="
  $P -u $R/cmp_floor.py $R/fl_base_L100.pt $R/fl_p7_L100.pt $K || true
  echo "==== within/cross, key=$K : base vs base2 (the CONTROL) ===="
  $P -u $R/cmp_floor.py $R/fl_base_L100.pt $R/fl_base2_L100.pt $K || true
done
