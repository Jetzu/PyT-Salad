"""N014 - reproduce, IN ONE PROCESS, the 2.72 A off-vs-P7 difference that the
cross-process floor job reported on a real mid-trajectory state.

The cross-process test is sound (its off-vs-off2 control came out inside the
floor), but a 2.72 A step from a 7.6e-06 pair perturbation is a DISCRETE
neighbour tie-flip, and a claim that big has to be reproduced first-hand before
anything is decided on it. Eager `forward_core`, both arms, same process, same
frozen state; `--reps` identical reruns give the floor at that same state.
"""
import argparse, sys
ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)
import torch                                                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--reps", type=int, default=12)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("highest")
from salad_torch import opt                                        # noqa: E402
from salad_torch.config import get_config                          # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict    # noqa: E402
from salad_torch.modules.diffusion import uniform_draw_count       # noqa: E402

dev = torch.device(a.device)
cfg = get_config(a.config)
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/%s-200k.npz" % a.config),
           cfg, strict=True)
st = torch.load(a.state, weights_only=False)
data = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in st["data"].items()}
prev = {k: v.to(dev) for k, v in st["prev"].items()}
print("loaded %s (upto=%s)  keys=%s" % (a.state, st.get("upto"), sorted(data)))
# The saved state has NO `uniform`, and `uniform=None` makes the neighbour
# selectors draw FRESH Gumbel noise on every call - which showed up as a 0.85 A
# "rerun floor" on CPU, where the model is deterministic and it must be 0. Pin
# it, and keep the CPU arm as the control that proves the input is frozen.
_L = int(data["pos"].shape[0])
_g = torch.Generator(device="cpu").manual_seed(1234)
data["uniform"] = torch.rand(uniform_draw_count(cfg), _L, _L,
                             generator=_g).clamp_(1e-6, 1 - 1e-6).to(dev)
print("pinned uniform %s" % (tuple(data["uniform"].shape),))


def run(flag):
    opt.BATCH_COND = flag
    out, _ = m(data, prev, predict_aa=True, uniform=data.get("uniform"),
               compute_distogram=False)
    return out


opt.BATCH_COND = False
ref = run(False)
ctls = [run(False) for _ in range(a.reps)]
got = run(True)
got2 = run(True)
opt.BATCH_COND = False
for k in ("pos", "local", "trajectory"):
    r = ref[k].double()
    floor = max((r - c[k].double()).abs().max().item() for c in ctls)
    d = (r - got[k].double()).abs().max().item()
    dp7 = (got[k].double() - got2[k].double()).abs().max().item()
    print("  %-11s rerun-floor(n=%d) %.3e   P7-vs-off %.3e   ratio %.2f   "
          "within-P7 %.3e" % (k, a.reps, floor, d, d / max(floor, 1e-30), dp7))
