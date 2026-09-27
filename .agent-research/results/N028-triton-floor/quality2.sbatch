#!/bin/bash -l
#SBATCH --job-name=n028_qual2
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor/logs/qual2_%j.log
set -u
# PLACEMENT: 1x H100. THE gate (tests/test_design_quality.py --n 160): it drives
# the COMPILED CUDA-graph path, which is the only path a parity test cannot see
# (K021). No CPU equivalent exists. ~3 GPU-h worst case, far under 25.
export PATH=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin:$PATH
P=/home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python
R=/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N028-triton-floor
# FREEZE the tree: copy the worktree into node-local /dev/shm at job start so a
# later edit in the worktree cannot be read half-written mid-run (the hazard
# N015's run_arms.sbatch documents), and so /hnvme's inode grace is untouched.
SRC=/home/hpc/v181bb/v181bb10/n028wt_o8
W=/dev/shm/n028tree_${SLURM_JOB_ID}
rm -rf $W; mkdir -p $W
cp -r $SRC/. $W/
cd $W
echo "frozen from HEAD: $(git -C $SRC rev-parse HEAD)"
md5sum salad_torch/modules/attention.py salad_torch/modules/blocks.py \
       salad_torch/geometry/features.py salad_torch/opt.py salad_torch/modules/fast.py
nvidia-smi --query-gpu=name --format=csv,noheader
OUT=/dev/shm/n028designs_${SLURM_JOB_ID}
mkdir -p $OUT

# THREE arms, one PROCESS each (opt.* is read at import time; TF32 is
# process-global), two seeds each pooled to n=160 - exactly N015's split.
#   base = both flags off (the shipping `fast` preset today)
#   o2   = N017/O2 IPA mul+sum          (reassociation, fast-preset candidate)
#   o6   = N028/O6 pair-feature merge   (bit-exact, exact-preset candidate)
for arm in base o8 all; do
  case $arm in
    base) export SALAD_OPT_IPAMULSUM=0 SALAD_OPT_PAIRFEAT=0 SALAD_OPT_FRAMEMULSUM=0;;
    o8)   export SALAD_OPT_IPAMULSUM=0 SALAD_OPT_PAIRFEAT=0 SALAD_OPT_FRAMEMULSUM=1;;
    all)  export SALAD_OPT_IPAMULSUM=1 SALAD_OPT_PAIRFEAT=1 SALAD_OPT_FRAMEMULSUM=1;;
  esac
  for spec in "42 40" "1234 120"; do
    set -- $spec
    SEED=$1; N=$2
    D=$OUT/${arm}_s${SEED}
    rm -rf $D
    export TORCHINDUCTOR_CACHE_DIR=/dev/shm/n028qc_${SLURM_JOB_ID}_${arm}
    echo "########## ARM $arm seed $SEED n $N (IPAMULSUM=$SALAD_OPT_IPAMULSUM PAIRFEAT=$SALAD_OPT_PAIRFEAT FRAMEMULSUM=$SALAD_OPT_FRAMEMULSUM)"
    $P -c "import sys; sys.path.insert(0,'$W'); import salad_torch.opt as o; print('FLAGCHECK IPA_MULSUM=',o.IPA_MULSUM,'PAIR_FEAT_MERGE=',o.PAIR_FEAT_MERGE,'FRAME_MULSUM=',o.FRAME_MULSUM)"
    $P tests/test_design_quality.py --n $N --preset fast --seed $SEED \
       --length 100 --out_path $D --keep
    echo "GATE_EXIT arm=$arm seed=$SEED rc=$?"
  done
  rm -rf /dev/shm/n028qc_${SLURM_JOB_ID}_${arm}
done

echo "########## POOLED n=160 STATS (N015 methodology)"
$P $R/stats.py \
   base=$OUT/base_s42,$OUT/base_s1234 \
   o8=$OUT/o8_s42,$OUT/o8_s1234 \
   all=$OUT/all_s42,$OUT/all_s1234 2>&1

echo "########## FLAG-ACTUALLY-FIRED CONTROL"
# If a flag never ran, its arm's PDBs are byte-identical to base at the same
# seed. O2 MUST differ (reassociation); O6 SHOULD be identical (bit-exact) -
# so O6's own fired-check is the CPU bitwise control, not this one.
for arm in o8 all; do
  for s in 42 1234; do
    same=0; tot=0
    for f in $OUT/base_s$s/*.pdb; do
      b=$(basename $f); o=$OUT/${arm}_s$s/$b
      [ -f "$o" ] || continue
      tot=$((tot+1)); cmp -s "$f" "$o" && same=$((same+1))
    done
    echo "FIREDCHECK arm=$arm seed=$s identical_pdbs=$same / $tot"
  done
done
cp $R/per_design.json $R/per_design_n160_o8.json 2>/dev/null || true
rm -rf $W $OUT
