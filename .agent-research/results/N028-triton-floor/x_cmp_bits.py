"""Compare two `forward_core` dumps bitwise. Exit 1 if any tensor differs."""
import sys
import torch
a = torch.load(sys.argv[1]); b = torch.load(sys.argv[2])
NAMES = ("pos", "local", "trajectory", "aa", "aa_features", "dssp",
         "distogram_neighbours", "raw_angles", "angles", "angle_pos", "aatype")
bad = 0
for i, (x, y) in enumerate(zip(a, b)):
    nm = NAMES[i] if i < len(NAMES) else f"out{i}"
    if not torch.is_tensor(x):
        continue
    same = torch.equal(x, y)
    d = (x.double() - y.double()).abs().max().item() if x.is_floating_point() else \
        float((x != y).sum())
    print(f"  {nm:<22} bitwise={same}  max|d|={d:.6e}")
    bad += (not same)
print(f"BITS {sys.argv[3] if len(sys.argv) > 3 else ''} "
      f"bit_identical={bad == 0}  differing_tensors={bad}")
sys.exit(1 if bad else 0)
