#!/bin/bash -l
#SBATCH --job-name=n028_o8d
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor/logs/o8diag_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
cd /home/hpc/v181bb/v181bb10/n028wt_o8
for arm in off on; do
  [ $arm = on ] && export SALAD_OPT_FRAMEMULSUM=1 || export SALAD_OPT_FRAMEMULSUM=0
  for C in default_ve_scaled multimotif_vp; do
    echo "######## FRAMEMULSUM=$SALAD_OPT_FRAMEMULSUM config=$C"
    SALAD_CONFIG=$C $P tests/test_model_parity.py 2>&1 | tail -30
  done
done
