"""N030 control: is the exact-filtered template selection BITWISE identical?

ONE process, several arms compiled back to back, so the cross-process codegen
variance N020 section 3c found (a no-op recompile of the SAME config differing
by 3.5e-05 A) cannot confound the comparison. `force_disable_caches` is
mandatory here: the FX-graph cache key does NOT see the monkeypatched candidate
filter, so with caches on the `exact` arm would silently be served the `all`
arm's compiled artifact and the comparison would be a tautology.

`base_rerun` is the methodology control: an arm byte-identical to `base`,
compiled last. If it is not 0.0 against `base`, nothing else in the table means
anything.

Run with SALAD_OPT_DETSEG=1: it removes the only source of CUDA run-to-run
nondeterminism in this model (the segment reduction's atomicAdd), which is what
makes a GPU bitwise comparison possible at all.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch._inductor.config as ind
import x_arm
from torch._functorch.compile_utils import fx_graph_cse

from x_common import NAMES, build  # noqa: E402

L = int(os.environ.get("N030_LEN", "100"))
ARMS = os.environ.get("N030_ARMS", "base,all,exact,base_rerun").split(",")

# Caches stay ON (disabling them makes every one of the ~300 GEMM sites
# re-autotune from scratch - measured: unusable). Cross-arm contamination is
# prevented the way the shipped code does it instead: the FX-graph cache key
# pickles `joint_custom_post_pass` BY NAME, so giving each arm its own
# identically-bodied, differently-NAMED pass salts the key. Without that an
# `exact` compile would be served the `all` arm's artifact and the comparison
# would be a tautology. It also makes `base_rerun` a genuine recompile.


def _cse0(graph):
    graph.owning_module.graph = fx_graph_cse(graph)


def _cse1(graph):
    graph.owning_module.graph = fx_graph_cse(graph)


def _cse2(graph):
    graph.owning_module.graph = fx_graph_cse(graph)


def _cse3(graph):
    graph.owning_module.graph = fx_graph_cse(graph)


def _cse4(graph):
    graph.owning_module.graph = fx_graph_cse(graph)


def _cse5(graph):
    graph.owning_module.graph = fx_graph_cse(graph)


_SALTS = [_cse0, _cse1, _cse2, _cse3, _cse4, _cse5]
ind.aggressive_fusion = True

from salad_torch import opt  # noqa: E402

assert opt.DET_SEGMENT, "run with SALAD_OPT_DETSEG=1 - this is the bitwise control"
torch.set_float32_matmul_precision("highest")
torch.set_grad_enabled(False)
torch.manual_seed(0)
m, cfg, args = build(L, device="cuda")

res = {}
meta = {}
for i, arm in enumerate(ARMS):
    real = "off" if arm.startswith("base") else arm
    info = x_arm.apply(real)
    ind.joint_custom_post_pass = _SALTS[i]
    torch._dynamo.reset()
    core = torch.compile(m.diffusion.forward_core, dynamic=False)
    out = core(*args, True, True, False)
    torch.cuda.synchronize()
    res[arm] = [o.detach().clone() if torch.is_tensor(o) else o for o in out]
    rep = x_arm.report(real)
    if rep is not None:
        rep.pop("per_shape", None)
    meta[arm] = {"info": info, "filter": rep, "counters": x_arm.counters()}
    print(f"[arm {arm}] compiled. {json.dumps(meta[arm])}", flush=True)

base = res[ARMS[0]]
table = {}
for arm in ARMS[1:]:
    row = {}
    ok = True
    for nm, a, b in zip(NAMES, base, res[arm]):
        if not torch.is_tensor(a):
            continue
        if a.dtype.is_floating_point:
            d = float((a.double() - b.double()).abs().max())
        else:
            d = float((a != b).sum())
        same = bool(torch.equal(a, b))
        ok &= same
        row[nm] = {"maxabs": d, "bitwise": same}
    table[arm] = {"bit_identical": ok, "outputs": row}
    print(f"[cmp {ARMS[0]} vs {arm}] bit_identical={ok} "
          f"pos={row['pos']['maxabs']:.3e} raw_angles={row['raw_angles']['maxabs']:.3e} "
          f"aatype_flips={row['aatype']['maxabs']:.0f}", flush=True)

print("RESULT " + json.dumps({"L": L, "arms": ARMS, "meta": meta, "cmp": table,
                              "torch": torch.__version__,
                              "gpu": torch.cuda.get_device_name(0)}), flush=True)
