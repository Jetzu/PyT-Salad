#!/bin/bash -l
#SBATCH --job-name=n025_mech
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /hnvme/workspace/v181bb10-salad/n025_x/job3_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
# node-local: /hnvme is inode-capped and already in grace, and an inductor
# cache is 4-7k files per process
export TMPWORK=/tmp/n025_$SLURM_JOB_ID
mkdir -p $TMPWORK/ind
export TORCHINDUCTOR_CACHE_DIR=$TMPWORK/ind
export TRITON_CACHE_DIR=$TMPWORK/triton
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
X=/hnvme/workspace/v181bb10-salad/n025_x
H=/hnvme/workspace/v181bb10-salad/n025
G=/hnvme/workspace/v181bb10-salad/n025_good
T=$X/traj; mkdir -p $T
N=16
nvidia-smi --query-gpu=name --format=csv,noheader
df -i /tmp | tail -1

echo "########## P1 PROBE: input sensitivity, HEAD compiled (fast preset)"
SALAD_ROOT=$H $P $X/x_probe.py --preset fast --out $T/probe_head_fast.json 2>&1 | tail -35
echo "########## P2 PROBE: GOOD compiled (fast preset)"
SALAD_ROOT=$G $P $X/x_probe.py --preset fast --out $T/probe_good_fast.json 2>&1 | tail -35
echo "########## P3 PROBE: HEAD EAGER"
SALAD_ROOT=$H $P $X/x_probe.py --eager --out $T/probe_head_eager.json 2>&1 | tail -35
echo "########## P4 PROBE: HEAD compiled, CSE VETOED"
N025_NOCSE=1 SALAD_ROOT=$H $P $X/x_probe.py --preset fast --out $T/probe_head_nocse.json 2>&1 | tail -35
echo "########## P5 PROBE: HEAD compiled, exact preset (fp32)"
SALAD_ROOT=$H $P $X/x_probe.py --preset exact --out $T/probe_head_exact.json 2>&1 | tail -35

echo "########## T1 TRAJ: HEAD --fast --preset fast, n=$N"
SALAD_ROOT=$H $P $X/x_traj.py --fast --preset fast --num_designs $N \
    --out $T/traj_head_fast.json 2>&1 | tail -24
echo "########## T2 TRAJ: GOOD --fast --preset fast, n=$N"
SALAD_ROOT=$G $P $X/x_traj.py --fast --preset fast --num_designs $N \
    --out $T/traj_good_fast.json 2>&1 | tail -24
echo "########## T3 TRAJ: HEAD --fast --preset fast, SALAD_OPT_DETSEG=1, n=$N"
SALAD_OPT_DETSEG=1 SALAD_ROOT=$H $P $X/x_traj.py --fast --preset fast --num_designs $N \
    --out $T/traj_head_fast_detseg.json 2>&1 | tail -24
echo "########## T4 TRAJ: HEAD EAGER, n=$N"
SALAD_ROOT=$H $P $X/x_traj.py --num_designs $N \
    --out $T/traj_head_eager.json 2>&1 | tail -24
rm -rf $TMPWORK
echo "ALL DONE"
