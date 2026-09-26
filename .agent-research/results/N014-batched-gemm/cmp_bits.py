"""Compare two dumps from x_cpu_bits.py / x_floor.py, bit for bit."""
import sys, torch
a, b = torch.load(sys.argv[1], weights_only=False), torch.load(sys.argv[2], weights_only=False)
print(f"A spec={a.get('spec', a.get('meta'))}\nB spec={b.get('spec', b.get('meta'))}")
ba, bb = a["blob"], b["blob"]
worst = 0.0
for k in sorted(ba):
    ta, tb = ba[k], bb.get(k)
    if ta is None or tb is None or not torch.is_tensor(ta):
        continue
    if ta.dtype.is_floating_point:
        d = (ta.double() - tb.double()).abs().max().item()
        worst = max(worst, d)
        print(f"  {k:<16} equal={torch.equal(ta, tb)!s:<5} max|d|={d:.6e}")
    else:
        print(f"  {k:<16} equal={torch.equal(ta, tb)!s:<5} (int)")
print(f"WORST {worst:.6e}")
