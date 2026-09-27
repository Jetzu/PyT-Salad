"""N030 smoke: does the bit-exact candidate filter work at all, on a toy graph?

Two `bmm`s at shapes the real model uses (a narrow projection, K=64, and a wide
one, K=400), compiled with `max_autotune_gemm`. Prints what the filter kept and
whether the compiled result is bitwise equal to eager cuBLAS.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch._inductor.config as ind
import x_arm

ARM = os.environ.get("N030_ARM", "exact")
ind.force_disable_caches = True
torch.set_float32_matmul_precision("highest")
torch.set_grad_enabled(False)
torch.manual_seed(0)

x1 = torch.randn(1, 4800, 64, device="cuda")
w1 = torch.randn(1, 64, 64, device="cuda")
x2 = torch.randn(1, 4800, 400, device="cuda")
w2 = torch.randn(1, 400, 64, device="cuda")


def f(x1, w1, x2, w2):
    return torch.bmm(x1, w1) + 1.0, torch.bmm(x2, w2) + 1.0


ref = f(x1, w1, x2, w2)
info = x_arm.apply(ARM)
print("arm info", info, flush=True)
g = torch.compile(f, dynamic=False)
got = g(x1, w1, x2, w2)
torch.cuda.synchronize()
eq = [bool(torch.equal(a, b)) for a, b in zip(ref, got)]
d = [float((a.double() - b.double()).abs().max()) for a, b in zip(ref, got)]
print("SMOKE " + json.dumps({"arm": ARM, "bitwise_vs_eager": eq, "maxabs": d,
                             "filter": x_arm.report(ARM)}), flush=True)
