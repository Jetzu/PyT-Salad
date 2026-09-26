"""N013 - the gate N009 section 7.7 asked for: is `aggressive_fusion=1`
bit-identical on the DETERMINISTIC backend?

CPU inductor has no atomics and no run-to-run noise, so a bit-comparison there
is a real test of whether a scheduler knob changed any element's arithmetic.
Inductor config is process-global and read at compile time, so the two arms
have to be two processes; this one dumps, `cmp_bits.py` compares.

    SALAD_IND=          python x_cpu_bits.py --out base.pt
    SALAD_IND=aggressive_fusion=1 python x_cpu_bits.py --out aggr.pt
"""
import argparse, os, sys
ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=int, default=64)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--eager", action="store_true", help="skip torch.compile")
a = ap.parse_args()

spec = os.environ.get("SALAD_IND", "")
if spec:
    import torch._inductor.config as ind
    for item in spec.split(","):
        if not item.strip():
            continue
        k, _, v = item.partition("=")
        k, v = k.strip(), v.strip()
        cur = getattr(ind, k)
        setattr(ind, k, type(cur)(int(v)) if isinstance(cur, bool) else
                (int(v) if isinstance(cur, int) else v))

torch.set_float32_matmul_precision("highest")
from salad_torch.config import get_config                       # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict  # noqa: E402
from salad_torch.modules.diffusion import uniform_draw_count     # noqa: E402

dev = torch.device(a.device)
cfg = get_config(a.config)
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"),
           cfg, strict=True)
D = m.diffusion
L, A = a.length, 5 + cfg.augment_size
g = torch.Generator(device="cpu").manual_seed(7)


def rnd(*sh):
    return torch.randn(*sh, generator=g).to(dev)


z = torch.zeros(L, dtype=torch.long, device=dev)
pos = rnd(L, A, 3) * 4
u = torch.rand(uniform_draw_count(cfg), L, L, generator=g).clamp_(1e-6, 1 - 1e-6).to(dev)
args = dict(pos_noised=pos, prev_pos=rnd(L, A, 3) * 4, prev_local=rnd(L, cfg.local_size),
            condition=rnd(L, cfg.local_size),
            dmap=torch.zeros(L, L, device=dev), omap=torch.zeros(L, L, 9, device=dev),
            dmap_mask=torch.zeros(L, L, dtype=torch.bool, device=dev),
            flags=torch.zeros(L, L, 3, device=dev),
            t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev),
            seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
            aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
            pos_gt=pos, resi=torch.arange(L, device=dev), chain=z, batch=z,
            mask=torch.ones(L, dtype=torch.bool, device=dev),
            cyclic_mask=None, uniform=u)
ORDER = ("pos_noised prev_pos prev_local condition dmap omap dmap_mask flags t_pos "
         "t_seq seq_noised aa_gt pos_gt resi chain batch mask cyclic_mask uniform").split()
pos_args = [args[k] for k in ORDER]

fn = D.forward_core if a.eager else torch.compile(D.forward_core, dynamic=False)
with torch.no_grad():
    if not a.eager:
        fn(*pos_args, True, False, False)          # warm the compile
    out = fn(*pos_args, True, False, False)        # need_aux, need_aa=False, pair_cond=False
    out_full = fn(*pos_args, True, True, False)
names = ("pos", "local", "trajectory")
blob = {n: t.cpu().clone() for n, t in zip(names, out[:3])}
blob.update({f"full_{i}": (t.cpu().clone() if torch.is_tensor(t) else None)
             for i, t in enumerate(out_full)})
torch.save({"spec": spec, "device": a.device, "length": L, "eager": a.eager,
            "blob": blob}, a.out)
print(f"saved {a.out}  spec={spec!r} device={a.device} eager={a.eager} "
      f"pos[0,1]={blob['pos'][0, 1].tolist()}")
