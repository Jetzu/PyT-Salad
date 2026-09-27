"""- make inductor's matmul TEMPLATES bit-exact and compile-invariant.

Background. Every `HkLinear` is an `einsum` that ATen lowers
to a batch-1 `bmm`, and the captured lean step at L=100 spends 62 % of its
GPU-busy time in 301 cuBLAS launches that average 5.98 us for arithmetic under
1 us - almost pure launch latency. Turning on inductor's matmul templates
(`max_autotune_gemm`) replaces 242 of those launches with Triton template
kernels and is worth -14 % of a lean step at L=100, the biggest single number
anyone has measured on this model.

rejected it anyway, for two reasons, and both are about SELECTION rather
than about the templates themselves:

* the winning kernel is chosen by a **timing race**, so at a shape where two
  candidates are within benchmarking noise two independent compiles of the same
  configuration can pick different winners - measured: three cold-cache
  compiles of one config disagreed by up to 8.87e-05 A, i.e. the bits depend on
  a race even on the same card; and
* the winner is only *approximately* equal to cuBLAS - whole-forward
  `pos` max|d| 5.84e-05 A at L=100 against a 2e-05 A parity floor.

This module changes the selection CRITERION from

    the fastest candidate

to

    the fastest candidate that is BITWISE IDENTICAL to the ATen (cuBLAS)
    kernel at this exact shape, measured on the machine that is compiling.

That answers both objections at once and it answers them *structurally*:

* bit-exactness: every survivor produces the same bits as the kernel the model
  runs today, so the whole forward is unchanged bit for bit;
* determinism: the timing race can now only choose between kernels that are
  bitwise equal, so its outcome cannot reach the output. Two compiles that pick
  different winners still produce the same bits.

Why measuring on random inputs is enough. Two fp32 GEMM kernels differ in their
results only if they differ in the ORDER (or the fusion) of their
multiply-accumulates, which is a property of the kernel, fixed at compile time
and independent of the data. So a single random draw at the real shape compares
those two orders across every element of the output at once (here: 4800x64 =
307k independent dot products of length K). If two orders differ at all, fp32
random data disagrees on essentially all of them; agreeing on all of them means
the orders coincide. The check is `torch.equal` on the raw bytes, not a
tolerance.

Why this is the PORTABLE form of the idea. The obvious alternative - ship a
per-shape allowlist, or a pinned per-shape config, measured on this H100 - is
exactly the kind of tuned constant this project may not ship: it is wrong on a
4090 and wrong again on sm_120, where cuBLAS picks different kernels and the
Triton candidate set is filtered differently by the device's shared memory. The
verification here is re-run by inductor at compile time on whatever card is
present, so a card where no Triton candidate is exact simply keeps today's
cuBLAS kernels and loses the speedup rather than the exactness. Nothing in this
file names an architecture, a block size, a warp count or a stage count; the
candidate list and the winner both still come from inductor.

Failure is fail-safe in every branch: any exception anywhere in the
verification falls back to the ATen choice alone, i.e. exactly the kernel the
model runs with templates off.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional

__all__ = ["install", "uninstall", "STATS", "report"]

#: filled in per shape so a job log can show what the filter did.
STATS: List[dict] = []

_ORIG: dict = {}
_MODE = "exact"
_STABLE_TOL = 0.02
_LOG = os.environ.get("SALAD_GEMM_EXACT_LOG", "1") not in ("0", "false", "no", "")


def _bitwise_equal(a, b) -> bool:
    """True iff `a` and `b` hold the same bits (not the same value)."""
    import torch

    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    x = a.contiguous().reshape(-1).view(torch.uint8)
    y = b.contiguous().reshape(-1).view(torch.uint8)
    return bool(torch.equal(x, y))


def _example_inputs(input_nodes, layout):
    """Mirror `AlgorithmSelectorCache.make_benchmark_fn.get_inputs`."""
    import torch
    from torch._inductor import config as icfg
    from torch._inductor.select_algorithm import AlgorithmSelectorCache as A
    from torch._inductor.virtualized import V

    uniq = {x.get_name(): A.benchmark_example_value(x) for x in input_nodes}
    tri_inputs = list(uniq.values())
    ext_inputs = [
        torch.as_strided(
            uniq[n.get_name()],
            V.graph.sizevars.size_hints(
                n.get_size(), fallback=icfg.unbacked_symint_fallback),
            V.graph.sizevars.size_hints(
                n.get_stride(), fallback=icfg.unbacked_symint_fallback),
            V.graph.sizevars.size_hint(
                n.get_layout().offset, fallback=icfg.unbacked_symint_fallback),
        )
        for n in input_nodes
    ]
    out = A.benchmark_example_value(layout)
    out_ext = torch.as_strided(
        out, out.size(), out.stride(),
        V.graph.sizevars.size_hint(layout.offset))
    return tri_inputs, ext_inputs, out, out_ext


def _k_of(name: str, input_nodes) -> Optional[int]:
    """The contraction length, or None if it cannot be read statically."""
    from torch._inductor.virtualized import V

    try:
        mat2 = input_nodes[-1]
        return int(V.graph.sizevars.size_hint(mat2.get_size()[-2]))
    except Exception:
        return None


def _filter(name: str, choices, input_nodes, layout, max_k: Optional[int]):
    """Keep the ATen choice plus every Triton choice that matches it bitwise."""
    import torch
    from torch._inductor.select_algorithm import TritonTemplateCaller

    aten = [c for c in choices if not isinstance(c, TritonTemplateCaller)]
    tri = [c for c in choices if isinstance(c, TritonTemplateCaller)]
    rec: dict[str, Any] = {"op": name, "n_aten": len(aten), "n_triton": len(tri)}
    if not aten or not tri:
        # nothing to verify against (or nothing to verify): leave untouched.
        rec["action"] = "untouched"
        STATS.append(rec)
        return choices

    k = _k_of(name, input_nodes)
    rec["K"] = k
    try:
        rec["shape"] = [list(map(int, n.get_size())) for n in input_nodes]
    except Exception:
        rec["shape"] = None
    if max_k is not None and (k is None or k > max_k):
        # a compile-time economy only: above `max_k` the template loses to
        # cuBLAS anyway on this model's wide projections , so do not
        # pay to compile and verify ~40 candidates that cannot win.
        rec.update(action="aten_only_by_K", n_exact=0)
        STATS.append(rec)
        return aten

    try:
        tri_inputs, ext_inputs, out, out_ext = _example_inputs(input_nodes, layout)
        if len(tri_inputs) != len(input_nodes):
            raise RuntimeError("deduplicated inputs - cannot verify safely")
        aten[0].benchmark(*ext_inputs, out=out_ext)
        ref = out_ext.clone()
    except Exception as e:  # fail-safe: today's kernel
        rec.update(action="aten_only_on_error", err=repr(e)[:200], n_exact=0)
        STATS.append(rec)
        return aten

    keep, errs = [], 0
    for c in tri:
        try:
            out.zero_()
            c.precompile()
            c.benchmark(*tri_inputs, out=out)
        except Exception:
            errs += 1
            continue
        if _bitwise_equal(out_ext, ref):
            keep.append(c)
    del ref, out, out_ext
    rec.update(action="verified", n_exact=len(keep), n_failed=errs)
    STATS.append(rec)
    if _LOG:
        print(f"[gemm-exact] {name} shape={rec['shape']} K={k}: "
              f"{len(keep)}/{len(tri)} triton candidates are bitwise equal to "
              f"cuBLAS ({errs} failed to compile/run)", flush=True)
    return aten + keep


def _stable_pick(name, choices, input_nodes, layout, tol: float):
    """Collapse the candidate list to ONE deterministically-chosen winner.

    the project notes, section 3d: `max_autotune_gemm`'s selection is a TIMING RACE, and at a
    shape where two candidates are within benchmarking noise two cold-cache
    compiles of the same configuration pick different winners - so the model's
    output bits depend on a race even on one card (measured: three compiles
    disagreed by up to 8.87e-05 A). This mode keeps the measurement (which is
    what makes the choice portable across cards) but makes the TIE-BREAK
    deterministic: everything within `tol` of the fastest candidate is a tie,
    and the tie is broken on `hash_key()` - a content hash of the generated
    kernel, which is the same on every machine and in every process.

    Handing back a single choice also short-circuits inductor's own selection
    (`AlgorithmSelectorCache.__call__` returns immediately at `len(choices)==1`),
    so no second race can follow.

    This does NOT make the result equal to cuBLAS - use `install(mode="exact")`
    for that. It makes it the SAME on every compile.
    """
    timed = []
    for c in choices:
        try:
            tri_inputs, ext_inputs, out, out_ext = _example_inputs(input_nodes, layout)
            c.precompile()
            from torch._inductor.select_algorithm import TritonTemplateCaller
            if isinstance(c, TritonTemplateCaller):
                t = c.benchmark(*tri_inputs, out=out)
            else:
                t = c.benchmark(*ext_inputs, out=out_ext)
        except Exception:
            continue
        if t == t and t < float("inf"):
            timed.append((float(t), str(c.hash_key()), c))
    if not timed:
        return choices
    best = min(t for t, _, _ in timed)
    tied = sorted([r for r in timed if r[0] <= best * (1.0 + tol)],
                  key=lambda r: r[1])
    STATS.append({"op": name, "action": "stable", "n_candidates": len(timed),
                  "n_tied": len(tied), "best_ms": best,
                  "winner": tied[0][1][:60]})
    if _LOG:
        print(f"[gemm-stable] {name}: {len(tied)}/{len(timed)} candidates within "
              f"{tol:.0%} of {best*1e3:.1f} us; picked {tied[0][1][:40]}", flush=True)
    return [tied[0][2]]


def _wrap(orig, max_k):
    def wrapper(name, choices, input_nodes, layout, *a, **kw):
        if name in ("mm", "bmm", "addmm", "baddbmm") and len(choices) > 1:
            try:
                if _MODE == "stable":
                    choices = _stable_pick(name, choices, input_nodes, layout,
                                           _STABLE_TOL)
                else:
                    choices = _filter(name, choices, input_nodes, layout, max_k)
            except Exception as e:  # never let the filter break a compile
                print(f"[gemm-exact] filter failed on {name}: {e!r} - "
                      f"falling back to UNFILTERED choices", flush=True)
        return orig(name, choices, input_nodes, layout, *a, **kw)

    return wrapper


def install(max_k: Optional[int] = None, mode: str = "exact",
            stable_tol: float = 0.02) -> None:
    """Patch inductor's GEMM lowerings to select only bit-exact candidates.

    `max_k` is a compile-time economy, not a correctness knob: shapes with a
    larger contraction length skip verification and keep ATen. `None` verifies
    every shape.

    Note the import sites: `torch._inductor.kernel.{mm,bmm}` do
    `from ..select_algorithm import autotune_select_algorithm`, so the name has
    to be rebound in those modules - patching `select_algorithm` itself would
    be a no-op.
    """
    # `torch._inductor.lowering` is what pulls `kernel/*` in, and importing
    # `kernel.bmm` directly before it raises a circular ImportError
    # (`kernel.bmm` does `from .mm import _is_static_problem` while `kernel.mm`
    # is still half-initialised). Import lowering first and the submodules are
    # already there.
    global _MODE, _STABLE_TOL
    if mode not in ("exact", "stable"):
        raise ValueError(f"mode must be exact|stable, got {mode!r}")
    _MODE, _STABLE_TOL = mode, stable_tol
    import torch._inductor.lowering  # noqa: F401
    import torch._inductor.kernel.bmm as _bmm
    import torch._inductor.kernel.mm as _mm

    for mod in (_mm, _bmm):
        if mod not in _ORIG:
            _ORIG[mod] = mod.autotune_select_algorithm
        mod.autotune_select_algorithm = _wrap(_ORIG[mod], max_k)
    STATS.clear()


def uninstall() -> None:
    for mod, fn in _ORIG.items():
        mod.autotune_select_algorithm = fn
    _ORIG.clear()


def report() -> dict:
    """Aggregate what the filter did, for a job log or a records file."""
    n = len(STATS)
    ver = [s for s in STATS if s.get("action") == "verified"]
    return {
        "shapes": n,
        "verified": len(ver),
        "aten_only_by_K": sum(1 for s in STATS if s.get("action") == "aten_only_by_K"),
        "aten_only_on_error": sum(
            1 for s in STATS if s.get("action") == "aten_only_on_error"),
        "untouched": sum(1 for s in STATS if s.get("action") == "untouched"),
        "shapes_with_an_exact_template": sum(1 for s in ver if s["n_exact"]),
        "triton_candidates": sum(s["n_triton"] for s in ver),
        "triton_candidates_exact": sum(s["n_exact"] for s in ver),
        "per_shape": STATS,
    }
