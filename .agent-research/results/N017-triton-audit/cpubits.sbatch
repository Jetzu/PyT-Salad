#!/bin/bash -l
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --job-name=n017_cpubits
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N017-triton-audit/cpubits_%j.log
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_TREE=/hnvme/workspace/v181bb10-salad/n017live
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N017-triton-audit
cd $SALAD_TREE
for L in 64 100; do
  N017_LEN=$L N017_IDXMEAN=0              N017_DUMP=/tmp/n017_base_$L.pt TORCHINDUCTOR_CACHE_DIR=/tmp/n017cb_${SLURM_JOB_ID}_b python -u $R/x_cpu_bits.py
  N017_LEN=$L N017_IDXMEAN=0 N017_CSE=1   N017_DUMP=/tmp/n017_cse_$L.pt  TORCHINDUCTOR_CACHE_DIR=/tmp/n017cb_${SLURM_JOB_ID}_c python -u $R/x_cpu_bits.py
  N017_LEN=$L N017_IDXMEAN=1              N017_DUMP=/tmp/n017_idxm_$L.pt TORCHINDUCTOR_CACHE_DIR=/tmp/n017cb_${SLURM_JOB_ID}_m python -u $R/x_cpu_bits.py
  N017_LEN=$L N017_IDXMEAN=0 N017_IPA=1   N017_DUMP=/tmp/n017_ipa_$L.pt  TORCHINDUCTOR_CACHE_DIR=/tmp/n017cb_${SLURM_JOB_ID}_i python -u $R/x_cpu_bits.py
  python - <<PY
import torch
base=torch.load("/tmp/n017_base_$L.pt")
for tag in ("cse","idxm","ipa"):
    o=torch.load(f"/tmp/n017_{tag}_$L.pt")
    ident=all(torch.equal(a,b) for a,b in zip(base,o) if torch.is_tensor(a))
    md=max((a.double()-b.double()).abs().max().item() for a,b in zip(base,o)
           if torch.is_tensor(a) and a.is_floating_point())
    print(f"CPUBITS L=$L {tag:5s} bit_identical={ident}  max|d|={md:.3e}", flush=True)
PY
done
