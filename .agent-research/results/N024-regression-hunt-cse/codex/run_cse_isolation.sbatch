#!/bin/bash -l
#SBATCH --job-name=n024-cse
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N024-regression-hunt-cse/codex/cse_%j.log
set -euo pipefail
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
export SALAD_OPT_DETSEG=1
ROOT=/home/hpc/v181bb/v181bb10/SALAD_torch
R=$ROOT/.agent-research/results/N024-regression-hunt-cse/codex
cd "$ROOT"
echo "node=$(hostname) commit=$(git rev-parse HEAD) date=$(date)"
for arm in on off; do
  export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/n024_inductor_${arm}_${SLURM_JOB_ID}"
  /home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python -u "$R/run_cse_isolation.py" "$arm" \
    --fast --preset fast --num_aa 100 --num_designs 32 --seed 42 \
    --params /hnvme/workspace/v181bb10-salad/default_vp-200k.npz --out_path "$R/pdb_$arm"
done
/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python - "$R" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, 'tests')
from test_design_quality import metrics
r=Path(sys.argv[1])
for arm in ('on','off'):
    rows=[metrics(str(p)) for p in sorted((r/f'pdb_{arm}').glob('*.pdb'))]
    ok=[x for x in rows if not x['diverged']]
    print(arm, 'n',len(rows),'diverged',sum(x['diverged'] for x in rows),
          'mean_Rg_ok',sum(x['rg'] for x in ok)/len(ok) if ok else None,
          'mean_CA_CA_ok',sum(x['ca_ca'] for x in ok)/len(ok) if ok else None)
PY
