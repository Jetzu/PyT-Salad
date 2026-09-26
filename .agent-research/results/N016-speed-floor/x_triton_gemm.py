"""N016/M8 - is cuBLAS's ~3 us fixed cost per launch reducible?

M1 measured the marginal cost of an EMPTY kernel replayed in a captured graph
at 1.166 us. M4 measured the model's smallest real GEMM, (4800,9)x(9,64) -
5.5 MFLOP, i.e. 0.08 us of arithmetic at peak - at 3.15 us. The ~2 us gap is
cuBLAS's own fixed cost, not hardware. If an inductor/Triton GEMM template
lands nearer 1.2 us at the same shapes, then a large part of the 2.17 ms the
GEMM stage costs is a LIBRARY choice, which is a soft (reachable) term, not a
hard one - and that distinction is the whole point of this item.

Compares, at the model's real shapes, graph-replayed:
  cublas   torch.mm
  triton   torch.compile(..., max_autotune_gemm_backends="TRITON")
"""
from __future__ import annotations

import torch
from torch._inductor import config as icfg

dev = torch.device("cuda")
SHAPES = [
    (4800, 9, 64, "pair2.rot     (4800,  9)x(  9,64)"),
    (6400, 9, 64, "pair1.rot     (6400,  9)x(  9,64)"),
    (4800, 64, 8, "attn2.bias    (4800, 64)x( 64,  8)"),
    (4800, 60, 64, "pair2.dir     (4800, 60)x( 60,64)"),
    (100, 128, 512, "update.linear ( 100,128)x(128,512)"),
    (100, 1024, 128, "proj_out      ( 100,1024)x(1024,128)"),
    (4800, 240, 64, "pair2.pairvec (4800,240)x(240,64)"),
    (4800, 400, 64, "pair2.dist    (4800,400)x(400,64)"),
    (6400, 400, 64, "pair1.dist    (6400,400)x(400,64)"),
]
NREP = 20


def ev(fn, reps=100, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / reps


def cap(build):
    build()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        build()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        build()
    return g


icfg.max_autotune_gemm_backends = "TRITON"
for prec in ("highest", "high"):
    torch.set_float32_matmul_precision(prec)
    for (M, K, N, label) in SHAPES:
        a = torch.randn(M, K, device=dev)
        w = torch.randn(K, N, device=dev)
        c = torch.empty(M, N, device=dev)
        us_cublas = ev(cap(lambda: [torch.mm(a, w, out=c) for _ in range(NREP)]).replay) / NREP
        torch._dynamo.reset()
        try:
            f = torch.compile(lambda x, y: x @ y, mode="max-autotune-no-cudagraphs",
                              dynamic=False)
            with torch.no_grad():
                for _ in range(3):
                    ref = f(a, w)
                us_triton = ev(cap(lambda: [f(a, w) for _ in range(NREP)]).replay) / NREP
                err = (ref - torch.mm(a, w)).abs().max().item()
        except Exception as exc:
            us_triton, err = float("nan"), float("nan")
            print(f"M8 triton failed {label}: {str(exc)[:120]}", flush=True)
        print(f"M8 {prec:<8} {label:<30} cublas {us_cublas:6.3f} us   "
              f"triton {us_triton:6.3f} us   ratio {us_cublas/us_triton:4.2f}x   "
              f"max|d| {err:.3e}", flush=True)
        del a, w, c
        torch.cuda.empty_cache()
torch.set_float32_matmul_precision("highest")
