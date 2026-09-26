#!/bin/bash -l
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --job-name=n017_cpubits2
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N017-triton-audit/cpubits2_%j.log
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n017live
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N017-triton-audit
cd $SALAD_TREE
L=100
N017_LEN=$L N017_IDXMEAN=0            N017_DUMP=/tmp/n017b_base.pt TORCHINDUCTOR_CACHE_DIR=/tmp/n017cc_${SLURM_JOB_ID}_b python -u $R/x_cpu_bits.py
N017_LEN=$L N017_IDXMEAN=0 N017_CSE=1 N017_DUMP=/tmp/n017b_cse.pt  TORCHINDUCTOR_CACHE_DIR=/tmp/n017cc_${SLURM_JOB_ID}_c python -u $R/x_cpu_bits.py
TORCHINDUCTOR_FREEZING=1 N017_LEN=$L N017_IDXMEAN=0 N017_DUMP=/tmp/n017b_frz.pt TORCHINDUCTOR_CACHE_DIR=/tmp/n017cc_${SLURM_JOB_ID}_f python -u $R/x_cpu_bits.py
python - <<'PY'
import torch
names=("pos","local","trajectory","aa","aa_features","dssp","distogram_neighbours",
       "raw_angles","angles","angle_pos","aatype")
base=torch.load("/tmp/n017b_base.pt")
for tag in ("cse","frz"):
    o=torch.load(f"/tmp/n017b_{tag}.pt")
    ident=all(torch.equal(a,b) for a,b in zip(base,o) if torch.is_tensor(a))
    print(f"CPUBITS2 L=100 {tag:4s} bit_identical={ident}")
    for n,a,b in zip(names,base,o):
        if torch.is_tensor(a) and a.is_floating_point() and not torch.equal(a,b):
            print(f"   {n:22s} max|d|={(a.double()-b.double()).abs().max().item():.3e}")
PY
