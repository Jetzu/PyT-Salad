#!/bin/bash -l
#SBATCH --job-name=n014_floor2
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:50:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N014-batched-gemm/logs/floor2_%j.log
set -x
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=.agent-research/results/N014-batched-gemm
md5sum salad_torch/modules/cond_prologue.py salad_torch/opt.py
E="SALAD_OPT_AGGRFUSE=0"
# ONE frozen mid-trajectory state; three rep-sets of 40 against it.
env $E SALAD_OPT_BATCHCOND=0 $P -u $R/x_floor.py --length 100 --reps 40 --input real --upto 250 \
    --state-save $R/st2_L100.pt --tag off  --save $R/f2_off_L100.pt   || true
env $E SALAD_OPT_BATCHCOND=0 $P -u $R/x_floor.py --length 100 --reps 40 --input real \
    --state-load $R/st2_L100.pt --tag off2 --save $R/f2_off2_L100.pt  || true
env $E SALAD_OPT_BATCHCOND=1 $P -u $R/x_floor.py --length 100 --reps 40 --input real \
    --state-load $R/st2_L100.pt --tag p7   --save $R/f2_p7_L100.pt    || true
for K in pos local trajectory; do
  echo "#### key=$K  CONTROL: off vs off2 (same code, two processes) ####"
  $P -u $R/cmp_floor.py $R/f2_off_L100.pt $R/f2_off2_L100.pt $K || true
  echo "#### key=$K  TEST: off vs p7 ####"
  $P -u $R/cmp_floor.py $R/f2_off_L100.pt $R/f2_p7_L100.pt $K || true
done
