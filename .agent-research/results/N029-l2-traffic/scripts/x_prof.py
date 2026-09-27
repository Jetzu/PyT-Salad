"""N029/D - per-kernel TIME census of the captured lean step at HEAD."""
import os, sys, json, collections, torch
ROOT = os.environ.get("SALAD_ROOT", "/hnvme/workspace/v181bb10-salad/n029-l2")
sys.path.insert(0, ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.fast import FastDenoiser
from salad_torch.modules.model import StructureDiffusionPredict
from torch.profiler import profile, ProfilerActivity
PRESET = os.environ.get("SALAD_PRESET", "exact")
torch.set_float32_matmul_precision("highest")
dev = torch.device("cuda"); cfg = get_config("default_vp")
L = int(os.environ.get("SALAD_L", "100"))
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)
A = 5 + cfg.augment_size
z = torch.zeros(L, dtype=torch.long, device=dev); pos = torch.randn(L, A, 3, device=dev) * 4
d = dict(pos=pos, pos_noised=pos, aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
         seq=torch.full((L,), 20, dtype=torch.long, device=dev),
         seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
         residue_index=torch.arange(L, device=dev), chain_index=z, batch_index=z,
         mask=torch.ones(L, dtype=torch.bool, device=dev),
         t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev))
p = {"pos": torch.zeros(L, A, 3, device=dev), "local": torch.zeros(L, cfg.local_size, device=dev)}
with torch.no_grad():
    f = FastDenoiser(m, L, device=dev, capture=False, manual_graph=False,
                     mode="default", preset=PRESET)
    f.warmup(pair_cond=False)
    for _ in range(10): f(d, p, need_aa=False)
    torch.cuda.synchronize()
    REP = 20
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(REP): f(d, p, need_aa=False)
        torch.cuda.synchronize()
tot = collections.Counter(); cnt = collections.Counter()
for e in pr.key_averages():
    if e.device_type.name != "CUDA" and getattr(e, "self_device_time_total", 0) == 0:
        continue
    t = getattr(e, "self_device_time_total", 0.0)
    if t <= 0: continue
    tot[e.key] += t/REP; cnt[e.key] += e.count/REP
rows = [(v, cnt[k], k) for k,v in tot.items()]
rows.sort(reverse=True)
def cls(n):
    l=n.lower()
    if "triton" in l: return "triton"
    if any(x in l for x in ("xmma","gemv","cutlass","splitk","cublas","gemm","internal::kernel")): return "cublas"
    if "memcpy" in l or "memset" in l: return "memcpy"
    return "other"
agg=collections.Counter(); an=collections.Counter()
for v,c,k in rows:
    agg[cls(k)]+=v; an[cls(k)]+=c
print(f"PRESET={PRESET} total GPU busy {sum(tot.values()):.1f} us over {sum(cnt.values()):.0f} kernels")
for k in agg: print(f"  {k:<8} {agg[k]:8.1f} us  {an[k]:6.0f} launches")
print(f"\n{'us':>9}{'n':>6}  kernel")
for v,c,k in rows[:60]:
    print(f"{v:9.2f}{c:6.0f}  {k[:110]}")
json.dump([[k,c,v] for v,c,k in rows], open(os.environ["N029_OUT"], "w"))
print("PROF-DONE", flush=True)
