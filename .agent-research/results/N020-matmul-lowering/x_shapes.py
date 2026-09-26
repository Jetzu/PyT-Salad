"""N020/S: enumerate every HkLinear GEMM shape a real forward actually runs.

Eager, on CPU - the shape list is device-independent. `SALAD_N020_LEN` picks L.
Prints one JSON line per distinct (M, K, N, bias) with a count and the call
sites, so the per-shape win table has an authoritative shape list instead of
N016's hand-picked nine.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from x_common import build

torch.set_grad_enabled(False)
L = int(os.environ.get("N020_LEN", "100"))
NEED_AA = os.environ.get("N020_NEEDAA", "0") == "1"

m, cfg, args = build(L)
from salad_torch.modules.basic import HkLinear

seen = collections.Counter()
names = collections.defaultdict(set)
for name, mod in m.named_modules():
    if isinstance(mod, HkLinear):
        mod._n020_name = name
orig = HkLinear.forward


def patched(self, x):
    key = (int(x.reshape(-1, x.shape[-1]).shape[0]), int(self.in_size),
           int(self.out_size), bool(self.use_bias))
    seen[key] += 1
    names[key].add(getattr(self, "_n020_name", "?"))
    return orig(self, x)


HkLinear.forward = patched
m.diffusion.forward_core(*args, True, NEED_AA, False)
HkLinear.forward = orig
rows = []
for (M, K, N, b), c in sorted(seen.items(), key=lambda kv: -kv[1] * kv[0][1] * kv[0][2]):
    rows.append(dict(M=M, K=K, N=N, bias=b, count=c,
                     sites=sorted(names[(M, K, N, b)])[:4]))
print("SHAPES " + json.dumps(dict(L=L, need_aa=NEED_AA, distinct=len(rows),
                                  calls=sum(seen.values()), rows=rows)), flush=True)
