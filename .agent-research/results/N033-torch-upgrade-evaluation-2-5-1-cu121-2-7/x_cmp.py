"""N030: bitwise-compare dumps written by x_dump.py."""
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from x_common import NAMES  # noqa: E402

ref = torch.load(sys.argv[1])
for other in sys.argv[2:]:
    got = torch.load(other)
    row, ok = {}, True
    for nm, a, b in zip(NAMES, ref, got):
        if not torch.is_tensor(a):
            continue
        same = bool(torch.equal(a, b))
        ok &= same
        row[nm] = {"bitwise": same,
                   "maxabs": float((a.double() - b.double()).abs().max())
                   if a.dtype.is_floating_point else float((a != b).sum())}
    print("CMP " + json.dumps({"ref": sys.argv[1].split("/")[-1],
                               "got": other.split("/")[-1],
                               "bit_identical": ok, "outputs": row}), flush=True)
