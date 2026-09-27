#!/bin/bash -l
#SBATCH --job-name=n025_ladder
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /hnvme/workspace/v181bb10-salad/n025_x/job6_%j.log
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
N=16
nvidia-smi --query-gpu=name --format=csv,noheader
# LADDER: eager -> plain compile (no cudagraph) -> compile + manual capture.
# Same preset, same seed, same 250/200 schedule in every rung.
echo "########## L1 HEAD  compile WITHOUT cuda-graph capture (preset fast)"
SALAD_ROOT=$H $P $X/x_traj.py --fast --preset fast --no_capture --num_designs $N \
    --out $T/lad_head_nocapture.json 2>&1 | tail -20
echo "########## L2 GOOD  compile WITHOUT cuda-graph capture (preset fast)"
SALAD_ROOT=$G $P $X/x_traj.py --fast --preset fast --no_capture --num_designs $N \
    --out $T/lad_good_nocapture.json 2>&1 | tail -20
echo "########## L3 HEAD  --fast, SALAD_OPT_AGGRFUSE=0"
SALAD_OPT_AGGRFUSE=0 SALAD_ROOT=$H $P $X/x_traj.py --fast --preset fast --num_designs $N \
    --out $T/lad_head_noaggr.json 2>&1 | tail -20
echo "########## L4 HEAD  --fast, SALAD_OPT_BATCHCOND=0 (P7 off)"
SALAD_OPT_BATCHCOND=0 SALAD_ROOT=$H $P $X/x_traj.py --fast --preset fast --num_designs $N \
    --out $T/lad_head_nop7.json 2>&1 | tail -20
echo "########## L5 HEAD  --fast, CSE vetoed + AGGRFUSE=0 + BATCHCOND=0 (everything off)"
N025_NOCSE_ENV=1 SALAD_OPT_AGGRFUSE=0 SALAD_OPT_BATCHCOND=0 SALAD_ROOT=$H \
    $P $X/x_traj.py --fast --preset fast --no_cse --num_designs $N \
    --out $T/lad_head_alloff.json 2>&1 | tail -20
echo "########## L6 HEAD  --fast, CSE vetoed only"
SALAD_ROOT=$H $P $X/x_traj.py --fast --preset fast --no_cse --num_designs $N \
    --out $T/lad_head_nocse.json 2>&1 | tail -20
rm -rf $TMPWORK
echo ALL DONE
