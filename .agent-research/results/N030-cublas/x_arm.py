"""N030: turn an arm name into an inductor GEMM-selection configuration.

An arm is a '+'-joined list of features:

  off        templates off - today's shipped behaviour, the bitwise reference
  all        `max_autotune_gemm` unfiltered = N020's rejected -14 % arm
  exact      the same candidate list, filtered to the candidates that are
             BITWISE identical to cuBLAS at that shape (N030)
  exactK<n>  the same, but shapes with K > n skip verification and keep cuBLAS
  decomp<n>  inductor's `decompose_mm_pass` with
             min_first_dimension_decomposition=<n>: a batched `bmm` whose batch
             is >= n and two of whose (m, n, k) are < 32 becomes
             `(a[...,None] * b[...,None,:,:]).sum(-2)`, i.e. a Triton kernel
             (1.06 us to exist) instead of a cuBLAS launch (5.98 us). At L=100
             ~90 of the 301 launches are 3x3 frame rotations that qualify.
"""
import os
import sys

TREE = os.environ["SALAD_TREE"]
if TREE not in sys.path:
    sys.path.insert(0, TREE)


def apply(arm: str):
    import torch._inductor.config as ind
    feats = [f for f in arm.split("+") if f]
    info = {"arm": arm, "feats": feats}

    tpl = [f for f in feats
           if f == "all" or f.startswith("exact") or f.startswith("stable")]
    ind.max_autotune_gemm = bool(tpl)
    if ind.max_autotune_gemm:
        ind.max_autotune_gemm_backends = "ATEN,TRITON"
    exact = [f for f in feats if f.startswith("exact")]
    stable = [f for f in feats if f.startswith("stable")]
    from salad_torch import gemm_exact
    gemm_exact.uninstall()
    if exact:
        f = exact[0]
        max_k = int(f[6:]) if f.lower().startswith("exactk") and f[6:] else None
        gemm_exact.install(max_k=max_k, mode="exact")
        info["max_k"] = max_k
    elif stable:
        tol = float(stable[0][6:]) / 100.0 if stable[0][6:] else 0.02
        gemm_exact.install(mode="stable", stable_tol=tol)
        info["stable_tol"] = tol

    # `noepi`: forbid inductor from fusing an epilogue INTO the matmul
    # template. The candidate filter verifies the GEMM KERNEL against cuBLAS,
    # but with `benchmark_epilogue_fusion` (default True) the kernel that
    # actually runs is template+epilogue, chosen by a second timing race, and
    # the epilogue's own codegen is not what was verified.
    if "noepi" in feats:
        ind.benchmark_epilogue_fusion = False
        ind.epilogue_fusion = False
    else:
        ind.benchmark_epilogue_fusion = True
        ind.epilogue_fusion = True
    info["epilogue_fusion"] = ind.epilogue_fusion

    dec = [f for f in feats if f.startswith("decomp")]
    opts = dict(ind.post_grad_fusion_options)
    opts.pop("decompose_mm_pass", None)
    if dec:
        n = int(dec[0][6:]) if dec[0][6:] else 32
        opts["decompose_mm_pass"] = {"min_first_dimension_decomposition": n,
                                     "max_other_dimention_decomposition": 32}
        info["decomp_min_batch"] = n
    ind.post_grad_fusion_options = opts
    # the pass module reads both thresholds at IMPORT time, so re-set them on
    # the module too in case it is already imported (it is, after one compile).
    try:
        import torch._inductor.fx_passes.decompose_mem_bound_mm as d
        if dec:
            d.min_first_dimension_decomposition = int(dec[0][6:]) if dec[0][6:] else 32
            d.max_other_dimention_decomposition = 32
    except Exception:
        pass
    info["max_autotune_gemm"] = ind.max_autotune_gemm
    info["post_grad_fusion_options"] = {k: v for k, v in opts.items()}
    return info


def report(arm: str):
    if "exact" not in arm and "stable" not in arm:
        return None
    from salad_torch import gemm_exact
    return gemm_exact.report()


def counters():
    from torch._dynamo.utils import counters
    return {k: v for k, v in counters["inductor"].items()
            if "decompose" in k or "bmm" in k}
