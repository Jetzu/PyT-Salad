#!/bin/bash -l
#SBATCH --job-name=n028A2_gpu
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o /home/hpc/v181bb/v181bb10/salad_jobs/n028_stageA2_gpu_%j.log
set -u
# PLACEMENT: 1x H100. Parity suite (default state) + n=160 fast quality gate
# (the compiled/cudagraph path a parity fixture cannot see) + n=64 exact
# quality + speed re-baseline at L=100. ~1 GPU-h.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind
SRC=/home/hpc/v181bb/v181bb10/SALAD_torch
W=/dev/shm/n028A2wt_${SLURM_JOB_ID}
rm -rf $W; mkdir -p $W
cp -r $SRC/. $W/
cd $W
echo "frozen from HEAD: $(git -C $SRC rev-parse HEAD)"
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
$P -c "import salad_torch.opt as o; print('PAIR_FEAT_MERGE=', o.PAIR_FEAT_MERGE)"
rc=0
for cfg in default_vp default_vp_scaled default_ve_scaled multimotif_vp; do
  echo "===== test_model_parity $cfg"; SALAD_CONFIG=$cfg $P tests/test_model_parity.py; e=$?; echo "EXIT $e : test_model_parity $cfg"; [ $e -ne 0 ] && rc=1
done
for t in geometry stack encoder sampling; do
  echo "===== test_${t}_parity"; $P tests/test_${t}_parity.py; e=$?; echo "EXIT $e : test_${t}_parity"; [ $e -ne 0 ] && rc=1
done
echo "===== quality n=160 fast"; $P tests/test_design_quality.py --n 160 --preset fast --out_path ${TMPDIR:-/dev/shm}/f1_${SLURM_JOB_ID}; e=$?; echo "EXIT $e : quality fast n=160"; [ $e -ne 0 ] && rc=1
echo "===== quality n=64 exact"; $P tests/test_design_quality.py --n 64 --preset exact --out_path ${TMPDIR:-/dev/shm}/f2_${SLURM_JOB_ID}; e=$?; echo "EXIT $e : quality exact n=64"; [ $e -ne 0 ] && rc=1
rm -rf ${TMPDIR:-/dev/shm}/f1_${SLURM_JOB_ID} ${TMPDIR:-/dev/shm}/f2_${SLURM_JOB_ID}
echo "===== speed re-baseline L=100"
for preset in exact fast; do
  for r in 1 2; do
    $P -u scripts/bench_one.py --length 100 --no-eager --preset $preset --tag headA2_${preset}_r$r 2>&1 | grep -E '^RESULT'
  done
done
echo "GPU_SUITE_EXIT=$rc"
rm -rf $W
