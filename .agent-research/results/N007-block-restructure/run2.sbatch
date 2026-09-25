#!/bin/bash -l
#SBATCH --job-name=n007_r2
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:55:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N007-block-restructure/logs/n007r2_%j.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
D=.agent-research/results/N007-block-restructure
F='grep -v UserWarning'
echo "######## PAIRSPLIT L=100"
timeout 900 $P -u $D/x_pairsplit.py --length 100 2>&1 | grep -v "UserWarning\|warnings.warn"
echo "######## E2E"
for v in base tables k48 tables+k48 tables+onehot+nopc+k48 tables+onehot+nopc+k48+noaa; do
  timeout 900 $P -u $D/x_e2e.py $v --length 100 --profile 2>&1 | grep -v "^W0\|UserWarning\|warnings.warn"
done
echo "######## DONE2"
