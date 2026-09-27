#!/bin/bash -l
#SBATCH --job-name=n25p7
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /hnvme/workspace/v181bb10-salad/n025_p7_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
T=${TMPDIR:-/dev/shm}
export TORCHINDUCTOR_CACHE_DIR=$T/ind
export TRITON_CACHE_DIR=$T/triton
X=/hnvme/workspace/v181bb10-salad/n025_tools
H=/hnvme/workspace/v181bb10-salad/n025
G=/hnvme/workspace/v181bb10-salad/n025_good
W=$T/p7; mkdir -p $W
PAR=/hnvme/workspace/v181bb10-salad/default_vp-200k.npz
echo "node=$(hostname) TMPDIR=$T head=$(git -C $H rev-parse --short HEAD)"
nvidia-smi --query-gpu=name --format=csv,noheader

echo "########## S0 dump a REAL mid-rollout state (step 30 of 250, prev_pos live)"
$P $X/x_p7.py --root $H --state $W/state.pt --dump 2>&1 | tail -4

echo "########## U1 UNIT: prologue vs six sequential pair_2, GPU, HEAD"
$P $X/x_p7.py --root $H --state $W/state.pt --unit 2>&1 | tail -14
echo "########## U2 UNIT: same on CPU (deterministic by construction)"
$P $X/x_p7.py --root $H --state $W/state.pt --unit --device cpu 2>&1 | tail -14

echo "########## A1 eager    P7 ON"
SALAD_OPT_BATCHCOND=1 $P $X/x_p7.py --root $H --state $W/state.pt --out $W/e_on.npz 2>&1 | tail -3
echo "########## A2 eager    P7 OFF"
SALAD_OPT_BATCHCOND=0 $P $X/x_p7.py --root $H --state $W/state.pt --out $W/e_off.npz 2>&1 | tail -3
echo "########## A3 compiled P7 ON"
SALAD_OPT_BATCHCOND=1 $P $X/x_p7.py --root $H --state $W/state.pt --compiled --out $W/c_on.npz 2>&1 | tail -3
echo "########## A4 compiled P7 OFF"
SALAD_OPT_BATCHCOND=0 $P $X/x_p7.py --root $H --state $W/state.pt --compiled --out $W/c_off.npz 2>&1 | tail -3
echo "########## A5 eager    P7 ON, c5ab7a0 has no P7 -> reference"
$P $X/x_p7.py --root $G --state $W/state.pt --out $W/g_e.npz 2>&1 | tail -3
echo "########## A6 compiled c5ab7a0"
$P $X/x_p7.py --root $G --state $W/state.pt --compiled --out $W/g_c.npz 2>&1 | tail -3

echo "########## COMPARISONS (fp32, DETSEG=1, identical inputs)"
$P $X/cmp_npz.py $W/e_on.npz  $W/e_off.npz "EAGER    P7on vs P7off"
$P $X/cmp_npz.py $W/c_on.npz  $W/c_off.npz "COMPILED P7on vs P7off"
$P $X/cmp_npz.py $W/c_off.npz $W/e_off.npz "P7off    compiled vs eager"
$P $X/cmp_npz.py $W/c_on.npz  $W/e_on.npz  "P7on     compiled vs eager"
$P $X/cmp_npz.py $W/g_e.npz   $W/e_off.npz "c5ab7a0 eager vs HEAD eager P7off"
$P $X/cmp_npz.py $W/g_c.npz   $W/c_off.npz "c5ab7a0 compiled vs HEAD compiled P7off"
$P $X/cmp_npz.py $W/g_c.npz   $W/g_e.npz   "c5ab7a0 compiled vs eager"

echo "########## Q  end-to-end quality under DETSEG=1, HEAD, fast preset, n=32"
OUT=$T/q; mkdir -p $OUT
qarm () {
  L=$1; shift
  rm -rf $OUT/$L; mkdir -p $OUT/$L
  cd $H
  $P scripts/generate.py --config default_vp --params $PAR --num_aa 100 \
     --num_designs 32 --seed 42 --fast --preset fast --out_path $OUT/$L 2>&1 | tail -2
  $P $X/metrics.py $OUT/$L $L
  rm -rf $OUT/$L
}
export SALAD_OPT_DETSEG=1
SALAD_OPT_BATCHCOND=1 qarm Q1_DETSEG_P7on
SALAD_OPT_BATCHCOND=0 qarm Q2_DETSEG_P7off
unset SALAD_OPT_DETSEG
echo N025_P7_DONE
