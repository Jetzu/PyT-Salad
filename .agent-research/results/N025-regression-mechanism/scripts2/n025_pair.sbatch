#!/bin/bash -l
#SBATCH --job-name=n25pair
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH -o /hnvme/workspace/v181bb10-salad/n025_pair_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
T=${TMPDIR:-/dev/shm}
export TORCHINDUCTOR_CACHE_DIR=$T/ind
export TRITON_CACHE_DIR=$T/triton
X=/hnvme/workspace/v181bb10-salad/n025_tools
H=/hnvme/workspace/v181bb10-salad/n025
W=$T/pair; mkdir -p $W
echo "node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
echo "########## arm P7 ON  (compiled, DETSEG, fp32)"
SALAD_OPT_BATCHCOND=1 $P $X/x_pair.py --root $H --out $W/on.npz 2>&1 | tail -5
echo "########## arm P7 OFF (compiled, DETSEG, fp32)"
SALAD_OPT_BATCHCOND=0 $P $X/x_pair.py --root $H --out $W/off.npz 2>&1 | tail -5
echo "########## arm P7 OFF replicate (the NOISE FLOOR of this comparison)"
SALAD_OPT_BATCHCOND=0 $P $X/x_pair.py --root $H --out $W/off2.npz 2>&1 | tail -5
echo "########## AMPLIFICATION: P7on vs P7off"
$P $X/cmp_pair.py $W/on.npz $W/off.npz
echo "########## FLOOR: P7off vs P7off (identical config, rerun)"
$P $X/cmp_pair.py $W/off.npz $W/off2.npz
echo N025_PAIR_DONE
