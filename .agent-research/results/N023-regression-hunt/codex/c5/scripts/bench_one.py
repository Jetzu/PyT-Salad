"""Benchmark ONE length in a fresh process - compiling several shapes in one
process degrades inductor's cudagraphs, which made earlier numbers look flat.

Reports three numbers:
  eager     the uncompiled reference
  full      one fast-path forward that also runs the AA decoder + angle head
            (what a reporting sampling step costs)
  lean      one fast-path forward with `need_aa=False` (P2) - what the other
            499 steps of a 500-step design cost

The `SALAD_OPT_*` environment switches (salad_torch/opt.py) select which
proposals are live, so one job can measure each separately in its own process.
"""
import argparse, os, sys, time, torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser
from salad_torch import opt
ap=argparse.ArgumentParser(); ap.add_argument("--length",type=int,default=100)
ap.add_argument("--config",default="default_vp"); ap.add_argument("--iters",type=int,default=30)
ap.add_argument("--preset",default="exact"); ap.add_argument("--tag",default="")
ap.add_argument("--no-eager",action="store_true")
a=ap.parse_args()

def _apply_inductor_env():
    """`SALAD_IND="combo_kernels=1,max_fusion_size=256"` - inductor knobs.

    Benchmark-only, and only for knobs that merge or enlarge kernels without
    changing any element's arithmetic (horizontal/vertical fusion). Anything
    that swaps a GEMM algorithm is NOT in this category and belongs to the
    fast preset, not the exact one.
    """
    spec = os.environ.get("SALAD_IND", "")
    if not spec:
        return ""
    import torch._inductor.config as ind
    for item in spec.split(","):
        if not item.strip():
            continue
        k, _, v = item.partition("=")
        k, v = k.strip(), v.strip()
        cur = getattr(ind, k)
        setattr(ind, k, type(cur)(int(v)) if isinstance(cur, bool) else
                (int(v) if isinstance(cur, int) else v))
    return spec

torch.set_float32_matmul_precision("highest")
_IND = _apply_inductor_env()
dev=torch.device("cuda"); cfg=get_config(a.config)
m=StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"), cfg, strict=True)
L=a.length; A=5+cfg.augment_size; z=torch.zeros(L,dtype=torch.long,device=dev)
pos=torch.randn(L,A,3,device=dev)*4
d=dict(pos=pos,pos_noised=pos,aa_gt=torch.full((L,),20,dtype=torch.long,device=dev),
  seq=torch.full((L,),20,dtype=torch.long,device=dev),seq_noised=torch.full((L,),20,dtype=torch.long,device=dev),
  residue_index=torch.arange(L,device=dev),chain_index=z,batch_index=z,
  mask=torch.ones(L,dtype=torch.bool,device=dev),cyclic_mask=torch.zeros(L,dtype=torch.bool,device=dev),
  t_pos=torch.full((L,),0.5,device=dev),t_seq=torch.ones(L,device=dev))
p={"pos":torch.zeros(L,A,3,device=dev),"local":torch.zeros(L,cfg.local_size,device=dev)}
def bench(fn,it,warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/it*1e3
with torch.no_grad():
    te = 0.0 if a.no_eager else bench(lambda: m(d,p,compute_distogram=False), a.iters)
    # single chain, no conditioning -> the P4/P5 gate will come out False;
    # the hint only decides which graphs are pre-built.
    t_warm=time.perf_counter()
    fast=FastDenoiser(m,L,device=dev,preset=a.preset).warmup(
        pair_cond=not opt.SKIP_PAIR_COND)
    t_warm=time.perf_counter()-t_warm
    tf=bench(lambda: fast(d,p), a.iters)
    tl=bench(lambda: fast(d,p,need_aa=False), a.iters) if fast.lean_steps else tf
    print(f"      gate: pair_cond_active={fast._pc_active}  graphs={sorted(fast._graphs)}")
flags=(f"GATHER={int(opt.GATHER_ONEHOT)} HOIST={int(opt.HOIST_NEIGHBOURS)} "
       f"NOPC={int(opt.SKIP_PAIR_COND)} AUX={int(opt.NEED_AUX)} IND=[{_IND}]")
print(f"RESULT {a.tag:<12} L={L:<5} {flags}  eager {te:7.2f}  full {tf:6.3f} ms  "
      f"lean {tl:6.3f} ms   500-step {(tf+499*tl)/1000:6.2f} s   "
      f"compile {t_warm:5.0f} s")
