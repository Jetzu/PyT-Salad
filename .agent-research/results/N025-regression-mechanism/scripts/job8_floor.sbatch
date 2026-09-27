#!/bin/bash -l
#SBATCH --job-name=n025_floor
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /hnvme/workspace/v181bb10-salad/n025_x/job8_%j.log
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
echo "########## F1 GOOD compiled, fast preset"
SALAD_ROOT=$G $P $X/x_floor.py --preset fast --out $T/floor_good_fast.json 2>&1 | tail -14
echo "########## F2 HEAD compiled, fast preset"
SALAD_ROOT=$H $P $X/x_floor.py --preset fast --out $T/floor_head_fast.json 2>&1 | tail -14
echo "########## F3 HEAD compiled, fast preset, CSE VETOED"
N025_NOCSE=1 SALAD_ROOT=$H $P $X/x_floor.py --preset fast --out $T/floor_head_nocse.json 2>&1 | tail -14
echo "########## F4 HEAD compiled, fast preset, AGGRFUSE=0"
SALAD_OPT_AGGRFUSE=0 SALAD_ROOT=$H $P $X/x_floor.py --preset fast --out $T/floor_head_noaggr.json 2>&1 | tail -14
echo "########## F5 HEAD compiled, fast preset, BATCHCOND=0"
SALAD_OPT_BATCHCOND=0 SALAD_ROOT=$H $P $X/x_floor.py --preset fast --out $T/floor_head_nop7.json 2>&1 | tail -14
echo "########## F6 HEAD compiled, fast preset, DETSEG=1"
SALAD_OPT_DETSEG=1 SALAD_ROOT=$H $P $X/x_floor.py --preset fast --out $T/floor_head_detseg.json 2>&1 | tail -14
echo "########## F7 GOOD EAGER"
SALAD_ROOT=$G $P $X/x_floor.py --eager --out $T/floor_good_eager.json 2>&1 | tail -14
echo "########## F8 HEAD EAGER"
SALAD_ROOT=$H $P $X/x_floor.py --eager --out $T/floor_head_eager.json 2>&1 | tail -14
rm -rf $TMPWORK
echo ALL DONE
