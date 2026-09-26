#!/bin/bash -l
#SBATCH --job-name=n021_equiv
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N021-packed-batching/logs/equiv_%j.log
set -u
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR:-/dev/shm}/ind-$SLURM_JOB_ID
T=/hnvme/workspace/v181bb10-salad/n021tree
W=${TMPDIR:-/dev/shm}/n021e-$SLURM_JOB_ID
mkdir -p $W
cd $T
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
echo "### node $(hostname) $(date)"
# GATE: --batch_size 1 must be TODAY'S PATH, byte for byte. Under
# SALAD_OPT_DETSEG=1 the forward is reproducible by construction (N013), so
# "byte-identical PDBs" is a real statement and not an atomics coincidence.
echo "===== A. --group_size 4 (N019's path, no --batch_size at all)"
SALAD_OPT_DETSEG=1 timeout 2000 $P scripts/generate.py --fast --preset fast \
  --num_aa 100 --num_designs 4 --seed 42 --group_size 4 --out_path $W/ref \
  --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz 2>&1 | tail -3
echo "===== B. --batch_size 1 --group_size 4"
SALAD_OPT_DETSEG=1 timeout 2000 $P scripts/generate.py --fast --preset fast \
  --num_aa 100 --num_designs 4 --seed 42 --group_size 4 --batch_size 1 \
  --out_path $W/b1 --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz 2>&1 | tail -3
echo "===== diff"
for f in $W/ref/*.pdb; do
  b=$W/b1/$(basename $f)
  if cmp -s "$f" "$b"; then echo "IDENTICAL $(basename $f)"; else echo "DIFFER    $(basename $f)"; fi
done
md5sum $W/ref/*.pdb $W/b1/*.pdb
echo "### done $(date)"
rm -rf $TORCHINDUCTOR_CACHE_DIR $W
