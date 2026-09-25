"""R04: Encoder degenerate-input stress (CPU)."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch")
torch.manual_seed(0)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_param_npz, load_encoder
from salad_torch.modules.encoder import Encoder

# --- red-team shim: restore the removed Rigid.__getitem__ so the REST of the
# --- encoder can be stressed. Source is NOT edited; this is test-local.
from salad_torch.geometry.rigid import Rigid as _R
def _gi(self, key):
    assert key == (slice(None), None)
    return self.unsqueeze_batch(1)
_R.__getitem__ = _gi
print("SHIM: Rigid.__getitem__ restored for the stress run")


cfg = get_config("default_vp")
enc = Encoder(cfg).eval()
p = load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz")
load_encoder(enc, p, cfg, strict=True)
print("encoder loaded strict=True")

def mk(L, kind="normal"):
    g = torch.Generator().manual_seed(1)
    if kind == "normal":
        bb = torch.randn(L, 5, 3, generator=g) * 3
        bb = bb + torch.arange(L, dtype=torch.float32)[:, None, None] * 3.8
    elif kind == "collinear":           # N, CA, C exactly on the x axis
        bb = torch.zeros(L, 5, 3)
        bb[:, 0, 0] = torch.arange(L, dtype=torch.float32) * 3.8 - 1.0
        bb[:, 1, 0] = torch.arange(L, dtype=torch.float32) * 3.8
        bb[:, 2, 0] = torch.arange(L, dtype=torch.float32) * 3.8 + 1.0
        bb[:, 3, 0] = torch.arange(L, dtype=torch.float32) * 3.8 + 2.0
        bb[:, 4, 0] = torch.arange(L, dtype=torch.float32) * 3.8 + 0.5
    elif kind == "identical":           # every residue at the SAME point
        bb = torch.zeros(L, 5, 3)
        bb[:, 0, 0] = -1.0; bb[:, 2, 0] = 1.0; bb[:, 3, 1] = 1.0; bb[:, 4, 2] = 1.0
    elif kind == "zeros":               # all atoms at the origin
        bb = torch.zeros(L, 5, 3)
    elif kind == "huge":
        bb = torch.randn(L, 5, 3, generator=g) * 1e6
    d = dict(pos=bb, residue_index=torch.arange(L), chain_index=torch.zeros(L, dtype=torch.long),
             batch_index=torch.zeros(L, dtype=torch.long), mask=torch.ones(L, dtype=torch.bool))
    return d

def chk(tag, fn):
    try:
        local, pos = fn()
        bad = []
        if not torch.isfinite(local).all(): bad.append(f"local NaN/Inf ({int((~torch.isfinite(local)).sum())})")
        if not torch.isfinite(pos).all():   bad.append(f"pos NaN/Inf ({int((~torch.isfinite(pos)).sum())})")
        mx = float(pos[torch.isfinite(pos)].abs().max()) if torch.isfinite(pos).any() else float("nan")
        print(f"[{'BAD ' if bad else 'ok  '}] {tag:<30} pos|max|={mx:.4g}  {'; '.join(bad)}")
        return local, pos
    except Exception as e:
        print(f"[CRASH] {tag:<30} {type(e).__name__}: {str(e)[:130]}")
        return None, None

with torch.no_grad():
    for L in (0, 1, 2, 3, 5, 47, 48, 49):
        chk(f"L={L} normal", lambda L=L: enc(mk(L)))
    for kind in ("collinear", "identical", "zeros", "huge"):
        chk(f"L=32 {kind}", lambda k=kind: enc(mk(32, k)))
    # all-masked
    d = mk(32); d["mask"] = torch.zeros(32, dtype=torch.bool)
    chk("L=32 all-masked", lambda: enc(d))
    # backbone 0:5 passthrough contract (checklist #4)
    d = mk(64)
    _, pos = enc(d)
    err = float((pos[:, :5] - d["pos"]).abs().max())
    print(f"[{'ok  ' if err < 1e-3 else 'BAD '}] backbone 0:5 passthrough     max|d| = {err:.3e} A")
    # duplicate/ambiguous neighbour ties: two chains at identical coords
    d = mk(32); d["pos"] = torch.cat([d["pos"][:16], d["pos"][:16]], 0)
    d["chain_index"] = torch.cat([torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long)])
    chk("L=32 duplicate chains", lambda: enc(d))

# gradients
print("\n-- gradients --")
d = mk(32)
d["pos"] = d["pos"].requires_grad_(True)
local, pos = enc(d)
pos.square().sum().backward()
n_none = sum(1 for q in enc.parameters() if q.grad is None)
n_nan = sum(1 for q in enc.parameters() if q.grad is not None and not torch.isfinite(q.grad).all())
n_zero = sum(1 for q in enc.parameters() if q.grad is not None and float(q.grad.abs().max()) == 0.0)
print(f"params {sum(1 for _ in enc.parameters())}  grad=None {n_none}  grad nonfinite {n_nan}  grad all-zero {n_zero}")
print(f"input grad finite: {bool(torch.isfinite(d['pos'].grad).all())}")
# collinear gradient
d = mk(32, "collinear"); d["pos"] = d["pos"].requires_grad_(True)
try:
    _, pos = enc(d); pos.square().sum().backward()
    print(f"collinear input grad finite: {bool(torch.isfinite(d['pos'].grad).all())}  "
          f"max {float(d['pos'].grad[torch.isfinite(d['pos'].grad)].abs().max()):.3g}")
except Exception as e:
    print("collinear backward:", type(e).__name__, e)
