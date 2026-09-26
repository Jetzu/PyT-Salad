"""Kernel census of the compiled forward, so the remaining time can be named.

Profiling a CAPTURED graph is useless - the replay is one opaque op - so this
runs the same compiled callable with `capture=False`. The kernel set is
identical; only the launch gaps differ, and those are what the capture removes.

    python x_kernels.py [--length 100] [--lean]
"""
import argparse, os, sys, collections, torch
ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"; sys.path.insert(0, ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser
from salad_torch import opt
from torch.profiler import profile, ProfilerActivity

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--config", default="default_vp")
a = ap.parse_args()
torch.set_float32_matmul_precision("highest")
dev = torch.device("cuda"); cfg = get_config(a.config); L = a.length
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"),
           cfg, strict=True)
A = 5 + cfg.augment_size
z = torch.zeros(L, dtype=torch.long, device=dev); pos = torch.randn(L, A, 3, device=dev) * 4
d = dict(pos=pos, pos_noised=pos, aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
         seq=torch.full((L,), 20, dtype=torch.long, device=dev),
         seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
         residue_index=torch.arange(L, device=dev), chain_index=z, batch_index=z,
         mask=torch.ones(L, dtype=torch.bool, device=dev),
         cyclic_mask=torch.zeros(L, dtype=torch.bool, device=dev),
         t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev))
p = {"pos": torch.zeros(L, A, 3, device=dev),
     "local": torch.zeros(L, cfg.local_size, device=dev)}

def group(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("gemm", "sgemm", "cutlass", "ampere_", "sm90", "sm80",
                            "gemv", "dot_kernel", "nn_", "nt_", "tn_")):
        return "GEMM (cuBLAS)"
    if "triton" in n:
        return "Triton (fused pointwise/reduction)"
    if "sort" in n or "radix" in n or "segmented" in n or "cub" in n:
        return "sort / argsort"
    if "index" in n or "gather" in n or "scatter" in n or "take" in n:
        return "gather / scatter"
    if "elementwise" in n or "vectorized" in n or "copy" in n or "fill" in n:
        return "ATen elementwise / copy"
    return "other"

print(f"opts: GATHER={int(opt.GATHER_ONEHOT)} HOIST={int(opt.HOIST_NEIGHBOURS)} "
      f"NOPC={int(opt.SKIP_PAIR_COND)} LEAN={int(opt.LEAN_STEPS)}")
with torch.no_grad():
    f = FastDenoiser(m, L, device=dev, capture=False, manual_graph=False,
                     mode="default").warmup(pair_cond=False)
    for need_aa in ((True, False) if f.lean_steps else (True,)):
        for _ in range(5):
            f(d, p, need_aa=need_aa)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                f(d, p, need_aa=need_aa)
            torch.cuda.synchronize()
        tot = collections.Counter(); cnt = collections.Counter()
        for e in prof.key_averages():
            if e.device_time_total <= 0 or e.device_type.name != "CUDA":
                continue
            g = group(e.key)
            tot[g] += e.device_time_total / 10 / 1000.0     # ms per forward
            cnt[g] += e.count // 10
        print(f"\n=== L={L}  need_aa={need_aa}  "
              f"{sum(cnt.values())} kernels, {sum(tot.values()):.3f} ms GPU-busy")
        for g, ms in tot.most_common():
            print(f"  {g:<38} {cnt[g]:>5} kernels  {ms:7.3f} ms  "
                  f"{ms/max(cnt[g],1)*1000:6.2f} us each")
        print("  -- top 12 individual kernels --")
        for e in sorted((e for e in prof.key_averages()
                         if e.device_time_total > 0 and e.device_type.name == "CUDA"),
                        key=lambda e: -e.device_time_total)[:12]:
            print(f"    {e.key[:64]:<64} {e.count//10:>4}x "
                  f"{e.device_time_total/10/1000:7.3f} ms")
