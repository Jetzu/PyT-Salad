"""Per-step divergence between two paired rollouts (x_pair.py dumps)."""
import sys
import numpy as np
A, B = np.load(sys.argv[1]), np.load(sys.argv[2])
pa, pb = A["pos"], B["pos"]            # (D, S, L, 3) CA coords
na, nb = A["nbr"], B["nbr"]            # (D, S, L, K) discrete neighbour index
print(f"shape {pa.shape}  designs={pa.shape[0]} steps={pa.shape[1]}")
D, S = pa.shape[0], pa.shape[1]
for d in range(D):
    dm = np.abs(pa[d] - pb[d]).reshape(S, -1).max(1)
    nd = (na[d] != nb[d]).reshape(S, -1).mean(1)

    def first(th):
        i = np.argmax(dm > th)
        return int(i) if (dm > th).any() else -1
    print(f"\n--- design {d}: first |dCA| > 1e-5 @{first(1e-5)}  1e-3 @{first(1e-3)} "
          f"1e-2 @{first(1e-2)}  0.1 @{first(0.1)}  1.0 @{first(1.0)}  "
          f"final {dm[-1]:.3f} A")
    print(f"    {'step':>5} {'max|dCA| A':>12} {'discrete nbr disagree':>22}")
    for s in list(range(0, min(24, S))) + list(range(24, S, max(1, S // 12))) + [S - 1]:
        print(f"    {s:>5} {dm[s]:>12.3e} {nd[s]:>21.1%}")
