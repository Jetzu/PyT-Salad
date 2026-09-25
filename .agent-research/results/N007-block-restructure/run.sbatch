#!/bin/bash -l
#SBATCH --job-name=n007_restruct
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:55:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N007-block-restructure/logs/n007_%j.log
cd /home/hpc/v181bb/v181bb10/SALAD_torch
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
D=.agent-research/results/N007-block-restructure

echo "######## DIAG"
$P -u $D/x_micro.py diag --length 100
$P -u $D/x_micro.py diag --length 400

echo "######## MICRO"
for e in pairfuse neigh ipa; do
  echo "---- $e L=100"; timeout 600 $P -u $D/x_micro.py $e --length 100
done
echo "---- neigh L=400"; timeout 600 $P -u $D/x_micro.py neigh --length 400

echo "######## E2E"
for v in base tables onehot nopc noaa tables+onehot+nopc; do
  timeout 900 $P -u $D/x_e2e.py $v --length 100 --profile 2>&1 | grep -v "^W0\|UserWarning\|warnings.warn"
done
echo "######## DONE"
