"""Compare two x_megacache.py --out dumps (lists of tensors) for equality."""
import sys
import torch

a = torch.load(sys.argv[1])
b = torch.load(sys.argv[2])
ok = len(a) == len(b)
maxd = 0.0
for i, (x, y) in enumerate(zip(a, b)):
    if x.dtype.is_floating_point:
        d = float((x.double() - y.double()).abs().max())
        maxd = max(maxd, d)
        same = torch.equal(x, y)
    else:
        d = float((x != y).sum())
        same = d == 0
    ok &= same
    print(f"  tensor[{i}] shape={tuple(x.shape)} dtype={x.dtype} bitwise={same} maxabs={d:.3e}")
print(f"CMP_MEGACACHE {sys.argv[1]} vs {sys.argv[2]}: bit_identical={ok} max_abs_over_all={maxd:.3e}")
