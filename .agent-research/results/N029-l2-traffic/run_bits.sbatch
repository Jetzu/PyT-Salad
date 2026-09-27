#!/bin/bash -l
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=02:00:00
#SBATCH --job-name=n029_bits
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N029-l2-traffic/logs/bits_%j.log
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n029-l2
export TORCHINDUCTOR_COMPILE_THREADS=1
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N029-l2-traffic
cd $SALAD_TREE
echo "### node $(hostname) $(date) 48 cores, CPU partition (no GPU: this control IS the CPU determinism argument)"
for L in 64 100; do
for ARM in 0 1; do
  N029_LEN=$L N029_CSE=1 SALAD_OPT_PAIRVEC=$ARM N029_DUMP=$TMPDIR/n029_${L}_$ARM.pt \
    TORCHINDUCTOR_CACHE_DIR=$TMPDIR/cc_${SLURM_JOB_ID}_${L}_$ARM \
    python -u $R/scripts/x_cpu_bits.py
  rm -rf $TMPDIR/cc_${SLURM_JOB_ID}_${L}_$ARM
done
python - <<PY
import torch
names=("pos","local","trajectory","aa","aa_features","dssp","distogram_neighbours",
       "raw_angles","angles","angle_pos","aatype")
a=torch.load("$TMPDIR/n029_${L}_0.pt"); b=torch.load("$TMPDIR/n029_${L}_1.pt")
ident=all(torch.equal(x,y) for x,y in zip(a,b) if torch.is_tensor(x))
print(f"N029 CPUBITS L=${L}  PAIRVEC 0 vs 1  bit_identical={ident}")
for n,x,y in zip(names,a,b):
    if torch.is_tensor(x):
        d = 0.0 if torch.equal(x,y) else (x.double()-y.double()).abs().max().item()
        print(f"   {n:22s} max|d|={d:.3e}")
PY
rm -f $TMPDIR/n029_${L}_0.pt $TMPDIR/n029_${L}_1.pt
done
echo "### done $(date)"
