#!/bin/bash -l
#SBATCH --job-name=n007_r3
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N007-block-restructure/logs/n007r3_%j.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
D=.agent-research/results/N007-block-restructure
for LEN in 100 400; do
  echo "######## BATCH6 L=$LEN"
  timeout 700 $P -u $D/x_batch6.py --length $LEN 2>&1 | grep -v "UserWarning\|warnings.warn\|^W0"
done
echo "######## DONE3"
