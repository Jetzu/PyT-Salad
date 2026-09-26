"""N020/T: per-shape, is an inductor Triton matmul template faster than cuBLAS,
and is it BIT-IDENTICAL?

Shapes are ENUMERATED from a real forward (x_shapes' monkeypatch) rather than
hand-picked, so the table covers every GEMM `HkLinear` actually issues at this
length. Three arms per shape, all graph-replayed (never an eager loop - an
eager loop measures ~6-8 us of CPU dispatch and reports every small GEMM at
"8.0 us" regardless of shape, N016):

  bmm     einsum("...k,ki->...i")  -> aten.bmm with batch=1   (the pre-N020 form)
  extern  torch.mm + bias add      -> cuBLAS + a pointwise kernel
  tpl     compiled, TRITON-only    -> one templated kernel, bias in the epilogue

`max|d|` is the templated result against the cuBLAS one, at the SAME precision.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch._inductor import config as icfg
from torch.profiler import ProfilerActivity, profile

torch.set_grad_enabled(False)
dev = torch.device("cuda")
NREP = 20
L = int(os.environ.get("N020_LEN", "100"))


def enumerate_shapes():
    """(M, K, N, bias) -> count, from a real eager forward at this length."""
    import collections

    from x_common import build
    m, cfg, args = build(L, device="cuda")
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
    m.diffusion.forward_core(*args, True, True, False)
    HkLinear.forward = orig
    del m, args
    torch.cuda.empty_cache()
    return seen, names


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


def cap(build_fn):
    build_fn()
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        build_fn()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        build_fn()
    return g


def top_kernels(graph, n=3):
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
    ev_ = [e for e in p.key_averages()
           if e.device_type.name == "CUDA" and e.device_time_total > 0]
    ev_.sort(key=lambda e: -e.device_time_total)
    return [(e.key[:48], round(e.count / 5, 1)) for e in ev_[:n]]


seen, names = enumerate_shapes()
shapes = sorted(seen.items(), key=lambda kv: (-kv[0][0] * kv[0][1] * kv[0][2] * kv[1]))
print(f"NSHAPES L={L} distinct={len(shapes)} calls={sum(seen.values())}", flush=True)

icfg.max_autotune_gemm_backends = "TRITON"
icfg.autotune_fallback_to_aten = False
PREC = os.environ.get("N020_PREC", "highest")
torch.set_float32_matmul_precision(PREC)
rows = []
for (M, K, N, bias), count in shapes:
    site = sorted(names[(M, K, N, bias)])[0]
    a = torch.randn(M, K, device=dev)
    w = torch.randn(K, N, device=dev)
    b = torch.randn(N, device=dev)
    ref_bmm = torch.einsum("...k,ki->...i", a, w) + (b if bias else 0)
    ref_mm = torch.mm(a, w) + (b if bias else 0)

    def f_bmm():
        for _ in range(NREP):
            torch.einsum("...k,ki->...i", a, w).add_(b) if bias else \
                torch.einsum("...k,ki->...i", a, w)

    def f_ext():
        for _ in range(NREP):
            torch.mm(a, w).add_(b) if bias else torch.mm(a, w)

    us_bmm = ev(cap(f_bmm).replay) / NREP
    us_ext = ev(cap(f_ext).replay) / NREP
    torch._dynamo.reset()
    kn = []
    try:
        if bias:
            fc = torch.compile(lambda x, y, z: torch.mm(x, y) + z,
                               mode="max-autotune-no-cudagraphs", dynamic=False)
            out = fc(a, w, b)
            g = cap(lambda: [fc(a, w, b) for _ in range(NREP)])
        else:
            fc = torch.compile(lambda x, y: torch.mm(x, y),
                               mode="max-autotune-no-cudagraphs", dynamic=False)
            out = fc(a, w)
            g = cap(lambda: [fc(a, w) for _ in range(NREP)])
        us_tpl = ev(g.replay) / NREP
        kn = top_kernels(g)
        err = (out.double() - ref_mm.double()).abs().max().item()
    except Exception as exc:
        us_tpl, err = float("nan"), float("nan")
        print(f"T tpl FAILED ({M},{K})x({K},{N}): {str(exc)[:150]}", flush=True)
    d_bmm_mm = (ref_bmm.double() - ref_mm.double()).abs().max().item()
    rows.append(dict(M=M, K=K, N=N, bias=bias, count=count, site=site,
                     us_bmm=us_bmm, us_extern=us_ext, us_tpl=us_tpl,
                     ratio_tpl_vs_extern=us_ext / us_tpl if us_tpl == us_tpl else None,
                     ratio_extern_vs_bmm=us_bmm / us_ext,
                     maxd_tpl_vs_cublas=err, maxd_bmm_vs_mm=d_bmm_mm,
                     tpl_kernels=kn))
    print(f"T {PREC:<8} {site[:34]:<34} ({M:5d},{K:4d})x({K:4d},{N:4d}) n={count:<3d} "
          f"bmm {us_bmm:6.3f}  mm {us_ext:6.3f}  tpl {us_tpl:6.3f}  "
          f"tpl/mm {us_ext / us_tpl if us_tpl == us_tpl else float('nan'):4.2f}x  "
          f"mm/bmm {us_bmm / us_ext:4.2f}x  max|d|tpl {err:.3e}  "
          f"max|d|bmm-mm {d_bmm_mm:.3e}  {kn[:1]}", flush=True)
    del a, w, b, ref_bmm, ref_mm
    torch.cuda.empty_cache()
print("PERSHAPE " + json.dumps(dict(L=L, prec=PREC, rows=rows)), flush=True)
