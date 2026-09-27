#!/bin/bash -l
#SBATCH --job-name=n028_gates
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor/logs/gates_%j.log
set -u
# PLACEMENT: 1x H100 (the parity fixtures are compared on GPU, as every prior
# item ran them). <1 GPU-h.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor
W=/home/hpc/v181bb/v181bb10/n028wt_o8
md5sum $W/salad_torch/geometry/frames.py $W/salad_torch/geometry/rigid.py \
       $W/salad_torch/geometry/features.py $W/salad_torch/modules/attention.py \
       $W/salad_torch/modules/blocks.py $W/salad_torch/opt.py
for arm in shipdefault o6 o8 o2; do
  case $arm in
    shipdefault) export SALAD_OPT_PAIRFEAT=1 SALAD_OPT_FRAMEMULSUM=0 SALAD_OPT_IPAMULSUM=0;;
    o6)          export SALAD_OPT_PAIRFEAT=1 SALAD_OPT_FRAMEMULSUM=0 SALAD_OPT_IPAMULSUM=0;;
    o8)          export SALAD_OPT_PAIRFEAT=0 SALAD_OPT_FRAMEMULSUM=1 SALAD_OPT_IPAMULSUM=0;;
    o2)          export SALAD_OPT_PAIRFEAT=0 SALAD_OPT_FRAMEMULSUM=0 SALAD_OPT_IPAMULSUM=1;;
  esac
  echo "################ PARITY SUITE arm=$arm PAIRFEAT=$SALAD_OPT_PAIRFEAT FRAMEMULSUM=$SALAD_OPT_FRAMEMULSUM IPAMULSUM=$SALAD_OPT_IPAMULSUM"
  bash $R/run_gates.sh $W
done
# and the flags OFF, as the control that the harness itself is clean
export SALAD_OPT_PAIRFEAT=0 SALAD_OPT_FRAMEMULSUM=0 SALAD_OPT_IPAMULSUM=0
echo "################ PARITY SUITE arm=alloff"
bash $R/run_gates.sh $W
