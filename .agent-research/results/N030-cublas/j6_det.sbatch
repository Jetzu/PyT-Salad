#!/bin/bash -l
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --job-name=n030_det
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas/j6_det_%j.log
# PLACEMENT: 1x H100, 16 cores, <=6 GPU-h. Cross-PROCESS determinism: three
# independent COLD-cache compiles of the same arm, each in its own python with
# its own inductor cache dir, compared bitwise. `base` gets the same treatment
# as the noise floor (N020 3c).
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n030
export TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1
export PYTHONHASHSEED=0
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N030-cublas
D=/dev/shm/n030det_$SLURM_JOB_ID
mkdir -p $D
cd $SALAD_TREE
md5sum salad_torch/gemm_exact.py
L=${N030_LEN:-100}
for ARM in ${N030_ARMS:-off exact all}; do
  for REP in 1 2 3; do
    echo "########## DUMP ARM=$ARM REP=$REP L=$L"
    SALAD_OPT_DETSEG=1 N030_LEN=$L N030_ARM=$ARM N030_DUMP=$D/${ARM}_$REP.pt \
      SALAD_GEMM_EXACT_LOG=0 TORCHINDUCTOR_CACHE_DIR=$D/ind_${ARM}_$REP \
      python -u $R/x_dump.py || true
    rm -rf $D/ind_${ARM}_$REP
  done
  echo "########## CMP $ARM"
  python -u $R/x_cmp.py $D/${ARM}_1.pt $D/${ARM}_2.pt $D/${ARM}_3.pt || true
done
echo "########## CMP off vs exact"
python -u $R/x_cmp.py $D/off_1.pt $D/exact_1.pt $D/exact_2.pt $D/all_1.pt || true
rm -rf $D
