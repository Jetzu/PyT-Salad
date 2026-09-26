"""Compare two forward_core dumps bitwise. `python x_cmp.py tag base.pt other.pt`"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from x_common import NAMES

tag, a_p, b_p = sys.argv[1], sys.argv[2], sys.argv[3]
a, b = torch.load(a_p), torch.load(b_p)
ident = all(torch.equal(x, y) for x, y in zip(a, b) if torch.is_tensor(x))
print(f"BITS {tag:28s} bit_identical={ident}")
for n, x, y in zip(NAMES, a, b):
    if torch.is_tensor(x) and not torch.equal(x, y):
        if x.is_floating_point():
            d = (x.double() - y.double()).abs().max().item()
            print(f"   {n:22s} max|d|={d:.6e}")
        else:
            print(f"   {n:22s} INT DIFF n={(x != y).sum().item()}")
