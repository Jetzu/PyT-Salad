"""N013 item 4 - what would a DETERMINISTIC segment reduction cost, and does
each candidate actually remove the nondeterminism?

`index_sum` -> `scatter_add` -> atomicAdd is the measured source (notes/04 s8).
Three candidates, all isolated here before anything touches the model:

  A  torch.use_deterministic_algorithms(True)   - does scatter_add even honour it?
  B  zeros().index_add_() under (A)             - torch ships a deterministic index_add
  C  one-hot matmul  (onehot @ x)               - deterministic by construction (cuBLAS),
                                                  no global mode, compile/graph-safe
"""
import argparse, os, sys, time
ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--feat", type=int, default=512)
a = ap.parse_args()
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("highest")
dev = torch.device("cuda")
N, C = a.length, a.feat
x = torch.randn(N, C, device=dev)
index = torch.zeros(N, dtype=torch.long, device=dev)          # one segment: the batch case
chain = (torch.arange(N, device=dev) * 3 // N).long()         # three segments: the chain case
ar = torch.arange(N, device=dev)


def scat(x, idx):
    out = torch.zeros_like(x)
    return out.scatter_add(0, idx[:, None].expand_as(x), x)


def iadd(x, idx):
    out = torch.zeros_like(x)
    out.index_add_(0, idx, x)
    return out


def onehot_mm(x, idx):
    oh = (idx.view(1, N) == ar.view(N, 1)).to(x.dtype)
    return oh @ x


def repro(fn, idx, reps):
    rs = [fn(x, idx) for _ in range(reps)]
    eq = all(torch.equal(rs[0], r) for r in rs[1:])
    d = max((rs[0] - r).abs().max().item() for r in rs[1:])
    return eq, d


def bench(fn, idx, it=300):
    for _ in range(20):
        fn(x, idx)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it):
        fn(x, idx)
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / it * 1e6


for label, idx in (("batch(1 seg)", index), ("chain(3 seg)", chain)):
    print(f"\n--- {label}, N={N} C={C}, nondeterministic mode ---")
    for name, fn in (("scatter_add", scat), ("index_add_", iadd), ("onehot_mm", onehot_mm)):
        try:
            eq, d = repro(fn, idx, a.reps)
            print(f"  {name:<12} reproducible={eq!s:<5} max|d|={d:.3e}  "
                  f"{bench(fn, idx):7.2f} us")
        except Exception as e:                                  # noqa: BLE001
            print(f"  {name:<12} RAISED {type(e).__name__}: {str(e)[:120]}")
    # agreement between the three (they are different summation orders)
    r = {n: f(x, idx) for n, f in (("scatter_add", scat), ("index_add_", iadd),
                                   ("onehot_mm", onehot_mm))}
    print(f"  scatter_add vs onehot_mm max|d| = "
          f"{(r['scatter_add'] - r['onehot_mm']).abs().max().item():.3e}")
    print(f"  scatter_add vs index_add_ max|d| = "
          f"{(r['scatter_add'] - r['index_add_']).abs().max().item():.3e}")

print("\n=== torch.use_deterministic_algorithms(True) ===")
torch.use_deterministic_algorithms(True, warn_only=True)
import warnings
for label, idx in (("batch(1 seg)", index), ("chain(3 seg)", chain)):
    print(f"--- {label} ---")
    for name, fn in (("scatter_add", scat), ("index_add_", iadd), ("onehot_mm", onehot_mm)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                eq, d = repro(fn, idx, a.reps)
                warned = [str(x.message)[:80] for x in w][:1]
                print(f"  {name:<12} reproducible={eq!s:<5} max|d|={d:.3e}  "
                      f"{bench(fn, idx):7.2f} us  warn={warned}")
            except Exception as e:                              # noqa: BLE001
                print(f"  {name:<12} RAISED {type(e).__name__}: {str(e)[:140]}")
print("\n=== strict (warn_only=False) ===")
torch.use_deterministic_algorithms(True, warn_only=False)
for name, fn in (("scatter_add", scat), ("index_add_", iadd), ("onehot_mm", onehot_mm)):
    try:
        fn(x, index)
        print(f"  {name:<12} OK under strict deterministic mode")
    except Exception as e:                                      # noqa: BLE001
        print(f"  {name:<12} RAISED {type(e).__name__}: {str(e)[:140]}")
