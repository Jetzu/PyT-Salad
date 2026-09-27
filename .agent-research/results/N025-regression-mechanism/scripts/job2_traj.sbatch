#!/bin/bash -l
#SBATCH --job-name=n025_traj
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /hnvme/workspace/v181bb10-salad/n025_x/job2_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=/hnvme/workspace/v181bb10-salad/n025_cache2
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
X=/hnvme/workspace/v181bb10-salad/n025_x
H=/hnvme/workspace/v181bb10-salad/n025
G=/hnvme/workspace/v181bb10-salad/n025_good
T=$X/traj; mkdir -p $T
N=16
nvidia-smi --query-gpu=name --format=csv,noheader

echo "########## PROBE: input sensitivity, HEAD compiled (fast preset)"
SALAD_ROOT=$H $P $X/x_probe.py --preset fast --out $T/probe_head_fast.json 2>&1 | tail -30
echo "########## PROBE: input sensitivity, GOOD compiled (fast preset)"
SALAD_ROOT=$G $P $X/x_probe.py --preset fast --out $T/probe_good_fast.json 2>&1 | tail -30
echo "########## PROBE: input sensitivity, HEAD EAGER"
SALAD_ROOT=$H $P $X/x_probe.py --eager --out $T/probe_head_eager.json 2>&1 | tail -30
echo "########## PROBE: input sensitivity, HEAD compiled, CSE VETOED"
N025_NOCSE=1 SALAD_ROOT=$H $P $X/x_probe.py --preset fast --out $T/probe_head_nocse.json 2>&1 | tail -30

echo "########## TRAJ: HEAD --fast --preset fast, DETSEG=1, n=$N"
SALAD_OPT_DETSEG=1 SALAD_ROOT=$H $P $X/x_traj.py --fast --preset fast --num_designs $N \
    --out $T/traj_head_fast_detseg.json 2>&1 | tail -22
echo "########## TRAJ: GOOD --fast --preset fast, n=$N (no DETSEG at c5ab7a0)"
SALAD_ROOT=$G $P $X/x_traj.py --fast --preset fast --num_designs $N \
    --out $T/traj_good_fast.json 2>&1 | tail -22
echo "########## TRAJ: HEAD --fast --preset fast, no DETSEG, n=$N"
SALAD_ROOT=$H $P $X/x_traj.py --fast --preset fast --num_designs $N \
    --out $T/traj_head_fast.json 2>&1 | tail -22
echo "ALL DONE"
