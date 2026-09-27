#!/bin/bash -l
#SBATCH --job-name=n25blk
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /hnvme/workspace/v181bb10-salad/n025_blk_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
T=${TMPDIR:-/dev/shm}
export TORCHINDUCTOR_CACHE_DIR=$T/ind
export TRITON_CACHE_DIR=$T/triton
X=/hnvme/workspace/v181bb10-salad/n025_tools
H=/hnvme/workspace/v181bb10-salad/n025
W=$T/blk; mkdir -p $W
echo "node=$(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
$P $X/x_p7b.py --root $H --state $W/state.pt --out /dev/null --dump 2>&1|tail -2
run () { L=$1; C=$2; shift 2; echo "--- $L: $* compiled=$C"; env "$@" $P $X/x_p7b.py --root $H $C \
  --state $W/state.pt --out $W/$L.npz 2>&1 | grep -E '^\[b\]|Error|Traceback'; }
run e_on   "" SALAD_OPT_BATCHCOND=1
run e_off  "" SALAD_OPT_BATCHCOND=0
run c_on   --compiled SALAD_OPT_BATCHCOND=1
run c_off  --compiled SALAD_OPT_BATCHCOND=0
echo "########## per-block localisation"
$P $X/cmp_blk.py $W/e_on.npz $W/e_off.npz  "EAGER    P7on vs P7off"
$P $X/cmp_blk.py $W/c_on.npz $W/c_off.npz  "COMPILED P7on vs P7off"
$P $X/cmp_blk.py $W/c_off.npz $W/e_off.npz "P7off compiled vs eager (FLOOR)"
$P $X/cmp_blk.py $W/c_on.npz $W/e_on.npz   "P7on  compiled vs eager"
echo "########## which inductor pass owns it (all COMPILED, P7 ON)"
for pr in aggr0 nopat noepi inline nofuse reduceov; do
  echo "--- probe $pr"
  SALAD_OPT_BATCHCOND=1 $P $X/x_p7b.py --root $H --state $W/state.pt --compiled \
     --probe $pr --out $W/p_$pr.npz 2>&1 | grep -E '^\[b\]|Error|Traceback' | tail -3
  [ -f $W/p_$pr.npz ] && $P $X/cmp_blk.py $W/p_$pr.npz $W/e_on.npz "probe=$pr vs EAGER P7on"
done
echo N025_BLK_DONE
