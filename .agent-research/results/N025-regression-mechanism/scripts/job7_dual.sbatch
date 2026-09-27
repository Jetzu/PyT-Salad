#!/bin/bash -l
#SBATCH --job-name=n025_dual
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /hnvme/workspace/v181bb10-salad/n025_x/job7_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TMPWORK=/tmp/n025_$SLURM_JOB_ID; mkdir -p $TMPWORK/ind
export TORCHINDUCTOR_CACHE_DIR=$TMPWORK/ind
export TRITON_CACHE_DIR=$TMPWORK/triton
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
X=/hnvme/workspace/v181bb10-salad/n025_x
H=/hnvme/workspace/v181bb10-salad/n025
G=/hnvme/workspace/v181bb10-salad/n025_good
T=$X/traj; mkdir -p $T
nvidia-smi --query-gpu=name --format=csv,noheader
echo "########## D1 HEAD  compiled-vs-eager per step, driven by COMPILED"
SALAD_ROOT=$H $P $X/x_dual.py --num_designs 3 --out $T/dual_head.json 2>&1 | tail -8
echo "########## D2 GOOD  compiled-vs-eager per step, driven by COMPILED"
SALAD_ROOT=$G $P $X/x_dual.py --num_designs 3 --out $T/dual_good.json 2>&1 | tail -8
echo "########## D3 HEAD  DETSEG=1 probe (does the floor collapse?)"
SALAD_OPT_DETSEG=1 SALAD_ROOT=$H $P $X/x_probe.py --preset fast --out $T/probe_head_detseg.json 2>&1 | tail -20
echo "########## D4 HEAD  AGGRFUSE=0 probe"
SALAD_OPT_AGGRFUSE=0 SALAD_ROOT=$H $P $X/x_probe.py --preset fast --out $T/probe_head_noaggr.json 2>&1 | tail -20
echo "########## D5 HEAD  BATCHCOND=0 probe"
SALAD_OPT_BATCHCOND=0 SALAD_ROOT=$H $P $X/x_probe.py --preset fast --out $T/probe_head_nop7.json 2>&1 | tail -20
rm -rf $TMPWORK
echo ALL DONE
