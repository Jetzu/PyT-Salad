#!/bin/bash -l
#SBATCH --job-name=n007_r4
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:55:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N007-block-restructure/logs/n007r4_%j.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
D=.agent-research/results/N007-block-restructure
echo "######## E2E"
for v in base tables onehot nopc k48 noaa tables+onehot+nopc+k48 tables+onehot+nopc+k48+noaa; do
  timeout 700 $P -u $D/x_e2e.py $v --length 100 --profile 2>&1 | grep -v "^W0\|UserWarning\|warnings.warn"
done
echo "######## DONE4"
