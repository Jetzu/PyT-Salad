#!/bin/bash -l
#SBATCH --job-name=n25fuse
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /hnvme/workspace/v181bb10-salad/n025_fuse_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
T=${TMPDIR:-/dev/shm}
export TORCHINDUCTOR_CACHE_DIR=$T/ind
export TRITON_CACHE_DIR=$T/triton
X=/hnvme/workspace/v181bb10-salad/n025_tools
H=/hnvme/workspace/v181bb10-salad/n025
W=$T/fuse; mkdir -p $W
echo "node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
$P $X/x_p7b.py --root $H --state $W/state.pt --out /dev/null --dump 2>&1|tail -2
SALAD_OPT_BATCHCOND=1 $P $X/x_p7b.py --root $H --state $W/state.pt --out $W/ref.npz 2>&1|tail -1
echo "########## EAGER P7on = reference. Probes below are all COMPILED, P7 ON."
for pr in fuse2 fuse4 fuse8 fuse16 fuse32 fuse64 nosplitred nopersred nomultikern; do
  echo "--- probe $pr"
  SALAD_OPT_BATCHCOND=1 $P $X/x_p7b.py --root $H --state $W/state.pt --compiled \
     --probe $pr --out $W/p_$pr.npz 2>&1 | grep -E '^\[b\]|Error|Traceback'|tail -2
  [ -f $W/p_$pr.npz ] && $P $X/cmp_blk.py $W/p_$pr.npz $W/ref.npz "probe=$pr vs EAGER"
done
echo N025_FUSE_DONE
