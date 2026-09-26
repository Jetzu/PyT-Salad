"""N016 - census of the SHIPPING lean step, at the level a lower bound needs.

Sections:
  P   parameters actually reachable on a lean step (trunk only, no AA heads)
  F   arithmetic: exact matmul FLOPs via FlopCounterMode + an elementwise
      element count from a dispatch-mode tally
  B   memory: naive unfused traffic, plus the IRREDUCIBLE HBM set
      (params + graph inputs + graph outputs) and the PEAK LIVE activation
      set, which decides whether the rest can stay in L2
  D   critical-path depth of the aten graph under an explicit, conservative
      "what forces a global barrier" classification.  This is the term that
      makes a kernel-count floor a BOUND rather than an average.
  K   kernel census of the real compiled forward (uncaptured, so the
      profiler can see individual kernels) for the requested preset

One preset, one length, one process.
"""
from __future__ import annotations

import argparse
import collections
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = "/hnvme/workspace/v181bb10-salad/n016-floor"
sys.path.insert(0, ROOT)

from salad_torch import opt                                    # noqa: E402
from salad_torch.config import get_config                      # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.fast import FastDenoiser, PRESETS, _CORE_ARGS      # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--preset", default="exact")
ap.add_argument("--sections", default="P,F,B,D,K")
a = ap.parse_args()
S = set(a.sections.split(","))

dev = torch.device("cuda")
cfg = get_config(a.config)
L = a.length
torch.set_float32_matmul_precision("high" if PRESETS[a.preset].get("tf32") else "highest")

m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"),
           cfg, strict=True)
print(f"CFG {a.config} L={L} preset={a.preset} local={cfg.local_size} pair={cfg.pair_size} "
      f"depth={cfg.diffusion_depth} heads={cfg.heads} key={cfg.key_size} "
      f"factor={cfg.factor} atoms={5+cfg.augment_size} tf32={torch.get_float32_matmul_precision()}",
      flush=True)

# ---------------------------------------------------------------- inputs
fast = FastDenoiser(m, L, device=dev, preset=a.preset)
static = fast._blank_inputs()
# real-ish geometry so the neighbour selection does something representative
static["pos_noised"] = torch.randn(L, 5 + cfg.augment_size, 3, device=dev) * 4
static["prev_pos"] = torch.randn(L, 5 + cfg.augment_size, 3, device=dev) * 4
static["pos_gt"] = static["prev_pos"].clone()
core = m.diffusion.forward_core
ARGS_LEAN = [static[k] for k in _CORE_ARGS] + [True, False, False]   # need_aux, need_aa, pair_cond
ARGS_FULL = [static[k] for k in _CORE_ARGS] + [True, True, False]


# -------------------------------------------------------------------- P
if "P" in S:
    def nb(mod):
        return sum(p.numel() for p in mod.parameters())
    tot = nb(m)
    d = m.diffusion
    parts = {}
    for name, sub in d.named_children():
        parts[name] = nb(sub)
    print(f"P total_model_params={tot} ({tot*4/2**20:.2f} MiB fp32)", flush=True)
    for k, v in sorted(parts.items(), key=lambda x: -x[1]):
        print(f"P   diffusion.{k:<28} {v:>10} ({v*4/2**20:7.2f} MiB)", flush=True)
    lean_dead = sum(parts.get(k, 0) for k in
                    ("aa_decoder", "dssp_decoder", "angle_head", "distogram_decoder"))
    print(f"P LEAN-STEP live params ~= {nb(d)-lean_dead} "
          f"({(nb(d)-lean_dead)*4/2**20:.2f} MiB); dead on a lean step {lean_dead} "
          f"({lean_dead*4/2**20:.2f} MiB)", flush=True)


# ------------------------------------------------------------- F and B
class Tally(TorchDispatchMode):
    """Per-op element and byte tally. `bytes` is INPUT+OUTPUT nbytes, i.e. the
    traffic an implementation with NO fusion and NO cache would move - an
    upper bound on traffic, not a lower one."""

    def __init__(self):
        self.ops = collections.Counter()
        self.cnt = collections.Counter()
        self.bytes = collections.Counter()
        self.out_elems = collections.Counter()
        self.shapes = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **(kwargs or {}))
        name = str(func)
        self.cnt[name] += 1
        nbytes = 0
        for t in torch.utils._pytree.tree_flatten((args, kwargs))[0]:
            if isinstance(t, torch.Tensor):
                nbytes += t.numel() * t.element_size()
        oe = 0
        for t in torch.utils._pytree.tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor):
                nbytes += t.numel() * t.element_size()
                oe += t.numel()
        self.bytes[name] += nbytes
        self.out_elems[name] += oe
        if name in ("aten.mm.default", "aten.bmm.default", "aten.addmm.default"):
            sh = tuple(tuple(t.shape) for t in args if isinstance(t, torch.Tensor))
            self.shapes[(name, sh)] += 1
        return out


if "F" in S or "B" in S:
    from torch.utils.flop_counter import FlopCounterMode
    for tag, ARGS in (("lean", ARGS_LEAN), ("full", ARGS_FULL)):
        with torch.no_grad():
            with FlopCounterMode(display=False) as fc:
                core(*ARGS)
            flops = fc.get_total_flops()
            t = Tally()
            with t:
                core(*ARGS)
        print(f"F {tag}: matmul_FLOPs={flops/1e9:.4f} GFLOP  "
              f"aten_ops={sum(t.cnt.values())}  "
              f"unfused_traffic={sum(t.bytes.values())/2**20:.1f} MiB  "
              f"output_elements={sum(t.out_elems.values())/1e6:.2f} M", flush=True)
        if tag == "lean":
            top = t.bytes.most_common(14)
            for k, v in top:
                print(f"B   {k:<44} x{t.cnt[k]:<5} {v/2**20:8.2f} MiB "
                      f"out_elems {t.out_elems[k]/1e6:6.2f} M", flush=True)
            print("F   --- mm/bmm shapes on a lean step ---", flush=True)
            for (nm, sh), c in sorted(t.shapes.items(), key=lambda x: -x[1])[:30]:
                print(f"F     {nm.split('.')[1]:<6} {str(sh):<44} x{c}", flush=True)

if "B" in S:
    in_bytes = sum(x.numel() * x.element_size()
                   for x in ARGS_LEAN if isinstance(x, torch.Tensor))
    with torch.no_grad():
        outs = core(*ARGS_LEAN)
    out_bytes = sum(x.numel() * x.element_size()
                    for x in outs if isinstance(x, torch.Tensor))
    print(f"B graph_input_bytes={in_bytes/2**20:.3f} MiB "
          f"graph_output_bytes={out_bytes/2**20:.3f} MiB", flush=True)
    print(f"B     (of the input, `uniform` alone is "
          f"{static['uniform'].numel()*4/2**20:.3f} MiB)", flush=True)


# -------------------------------------------------------------------- D
# Critical-path depth of the ATEN graph, under an explicit and deliberately
# CONSERVATIVE classification of what forces a global barrier.
BARRIER = {
    # reductions: the output element depends on many input elements
    "aten.mm", "aten.bmm", "aten.addmm", "aten.baddbmm", "aten.matmul",
    "aten._softmax", "aten._log_softmax", "aten.softmax",
    "aten.native_layer_norm", "aten.layer_norm",
    "aten.sum", "aten.mean", "aten.amax", "aten.amin", "aten.max", "aten.min",
    "aten.var", "aten.var_mean", "aten.std", "aten.prod", "aten.cumsum",
    "aten.linalg_vector_norm", "aten.norm",
    # data movement whose output order is not the input order
    "aten.sort", "aten.argsort", "aten.topk", "aten.scatter_add",
    "aten.index_put", "aten.index_put_", "aten.scatter", "aten.scatter_reduce",
    "aten._unique", "aten.nonzero", "aten.unique_consecutive",
}
# everything else (pointwise, views, gathers, cat, cast, broadcast) is treated
# as FUSIBLE INTO A NEIGHBOUR, i.e. costs 0 extra barriers. That is generous to
# the implementation and therefore safe for a LOWER bound.

if "D" in S:
    from torch._dynamo.backends.common import aot_autograd

    captured = []

    def fw(gm, example_inputs):
        captured.append(gm)
        return gm.forward

    backend = aot_autograd(fw_compiler=fw)
    torch._dynamo.reset()
    fn = torch.compile(core, backend=backend, dynamic=False)
    try:
        with torch.no_grad():
            fn(*ARGS_LEAN)
    except Exception as exc:
        # the aten graph is captured by `fw` BEFORE it is executed; running the
        # traced module can fail on a view/stride detail that inductor would
        # have handled. The graph is all this section needs.
        print(f"D note: traced graph captured, execution of the raw fx module "
              f"raised ({type(exc).__name__}: {str(exc)[:80]}) - harmless here",
              flush=True)
    if not captured:
        print("D FAILED: no aten graph captured", flush=True)
    else:
        gm = captured[0]
        g = gm.graph
        lvl_b, lvl_all = {}, {}
        hist = collections.Counter()
        barrier_nodes = collections.Counter()
        for n in g.nodes:
            preds = [x for x in n.all_input_nodes]
            base_b = max([lvl_b.get(p, 0) for p in preds], default=0)
            base_a = max([lvl_all.get(p, 0) for p in preds], default=0)
            if n.op == "call_function":
                nm = str(n.target)
                short = ".".join(nm.split(".")[:2]) if nm.startswith("aten.") else nm
                hist[short] += 1
                is_bar = short in BARRIER
                if is_bar:
                    barrier_nodes[short] += 1
                lvl_b[n] = base_b + (1 if is_bar else 0)
                lvl_all[n] = base_a + 1
            else:
                lvl_b[n] = base_b
                lvl_all[n] = base_a
        D = max(lvl_b.values())
        Dall = max(lvl_all.values())
        # ---- weighted critical path -------------------------------------
        # Each barrier stage costs at least max(c_launch, its own arithmetic
        # at the hardware's PEAK rate). c_launch is measured (x_micro.py M1:
        # 1.166 us marginal replay cost of an empty kernel in a serial chain
        # inside a captured CUDA graph). Peak is the fp32 FMA peak of this
        # node (132 SM x 128 x 2 x 1.98 GHz = 66.91 TFLOP/s); the TF32
        # tensor-core column uses NVIDIA's 494.7 TFLOP/s dense figure.
        C_LAUNCH_US = 1.166
        PEAK_FP32 = 66.91e12
        PEAK_TF32 = 494.7e12

        def node_flops(n):
            if n.op != "call_function":
                return 0.0
            nm = str(n.target)
            if not (nm.startswith("aten.bmm") or nm.startswith("aten.mm")
                    or nm.startswith("aten.addmm")):
                return 0.0
            shapes = []
            for x in n.all_input_nodes:
                v = x.meta.get("val", None)
                if v is not None and hasattr(v, "shape"):
                    shapes.append(tuple(int(d) for d in v.shape))
            mats = [sh for sh in shapes if len(sh) in (2, 3)]
            if len(mats) < 2:
                return 0.0
            a, b = mats[-2], mats[-1]
            try:
                if len(a) == 3 and len(b) == 3:
                    return 2.0 * a[0] * a[1] * a[2] * b[2]
                if len(a) == 2 and len(b) == 2:
                    return 2.0 * a[0] * a[1] * b[1]
            except Exception:
                return 0.0
            return 0.0

        for peak, tag in ((PEAK_FP32, "fp32"), (PEAK_TF32, "tf32")):
            wl = {}
            for n in g.nodes:
                base = max([wl.get(pn, 0.0) for pn in n.all_input_nodes], default=0.0)
                w = 0.0
                if n.op == "call_function":
                    short = ".".join(str(n.target).split(".")[:2])
                    if short in BARRIER:
                        w = max(C_LAUNCH_US, node_flops(n) / peak * 1e6)
                wl[n] = base + w
            print(f"D weighted_critical_path_{tag} = {max(wl.values())/1000:.4f} ms "
                  f"(sum over the longest chain of max(c_launch={C_LAUNCH_US} us, "
                  f"node_flops/{peak/1e12:.1f} TFLOPs))", flush=True)
        # ---- NON-MATMUL arithmetic (red-team item R7) -------------------
        # FlopCounterMode counts matmuls only. Count the pointwise/reduction
        # work directly off the traced graph: output elements x a per-element
        # FLOP charge (1 for add/mul/sub/div-class, 4 for transcendentals and
        # rsqrt/sqrt/pow, which issue on the SFU at roughly 1/4 rate).
        CHEAP = 1.0
        DEAR = 4.0
        DEAR_OPS = ("exp", "log", "sqrt", "rsqrt", "pow", "erf", "tanh", "sigmoid",
                    "reciprocal", "sin", "cos", "gelu", "softmax")
        FREE_OPS = ("view", "permute", "unsqueeze", "squeeze", "slice", "expand",
                    "reshape", "t.default", "transpose", "detach", "select",
                    "as_strided", "_unsafe_view", "alias", "contiguous", "clone",
                    "empty", "full", "arange", "zeros", "ones", "lift_fresh",
                    "copy", "cat", "stack", "to.", "_to_copy", "index", "gather",
                    "sort", "argsort", "topk", "eq", "ne", "lt", "gt", "le", "ge",
                    "logical", "bitwise", "where", "masked", "any", "all")
        nm_flops = 0.0
        nm_elems = 0.0
        for n in g.nodes:
            if n.op != "call_function":
                continue
            nm = str(n.target)
            if nm.startswith(("aten.bmm", "aten.mm", "aten.addmm", "aten.baddbmm")):
                continue
            if any(f in nm for f in FREE_OPS):
                continue
            v = n.meta.get("val", None)
            if v is None or not hasattr(v, "numel"):
                continue
            ne = int(v.numel())
            nm_elems += ne
            nm_flops += ne * (DEAR if any(d in nm for d in DEAR_OPS) else CHEAP)
        print(f"D NON-MATMUL arithmetic: {nm_elems/1e6:.2f} M output elements, "
              f"{nm_flops/1e9:.4f} GFLOP charged -> {nm_flops/66.91e12*1e3:.4f} ms "
              f"at fp32 peak (matmul was {tot_fl_pre/1e9:.4f} GFLOP)"
              if False else
              f"D NON-MATMUL arithmetic: {nm_elems/1e6:.2f} M output elements, "
              f"{nm_flops/1e9:.4f} GFLOP charged -> {nm_flops/66.91e12*1e3:.4f} ms "
              f"at fp32 peak", flush=True)
        # ---- MINIMUM MATERIALISED TRAFFIC (red-team item R2) -------------
        # Under perfect pointwise fusion, a tensor only has to be written to
        # (and read back from) the memory hierarchy if it is produced or
        # consumed by a BARRIER op - a GEMM, a reduction, a sort. Sum those
        # tensors' bytes: that is a lower bound on traffic for ANY kernel-level
        # implementation of this dataflow, and it is what must be compared
        # against L2 (measured 6.53 TB/s) and HBM (measured 2.94 TB/s).
        seen = set()
        tb = 0
        for n in g.nodes:
            if n.op != "call_function":
                continue
            short = ".".join(str(n.target).split(".")[:2])
            if short not in BARRIER:
                continue
            group = list(n.all_input_nodes) + [n]
            for x in group:
                if id(x) in seen:
                    continue
                v = x.meta.get("val", None)
                if v is None or not hasattr(v, "numel"):
                    continue
                seen.add(id(x))
                tb += int(v.numel()) * v.element_size()
        print(f"D MIN-MATERIALISED-TRAFFIC (barrier-op inputs+outputs, each "
              f"counted once): {tb/2**20:.1f} MiB -> {tb/6.53e12*1e3:.4f} ms at "
              f"the measured 6.53 TB/s L2 rate, {tb/2.94e12*1e3:.4f} ms at the "
              f"measured 2.94 TB/s HBM rate", flush=True)
        nbmm = sum(1 for n in g.nodes if n.op == "call_function"
                   and str(n.target).startswith(("aten.bmm", "aten.mm", "aten.addmm")))
        nbmm_fl = sum(1 for n in g.nodes if node_flops(n) > 0)
        print(f"D flop-annotation coverage: {nbmm_fl}/{nbmm} matmul nodes "
              f"(0/N means node.meta['val'] was absent and the weighted path "
              f"degenerates to D x c_launch)", flush=True)
        tot_fl = sum(node_flops(n) for n in g.nodes)
        print(f"D graph_matmul_flops={tot_fl/1e9:.4f} GFLOP "
              f"-> {tot_fl/PEAK_FP32*1e3:.4f} ms at fp32 peak, "
              f"{tot_fl/PEAK_TF32*1e3:.4f} ms at tf32 peak", flush=True)
        # ---- a fusion-relaxed depth -------------------------------------
        # collapse runs of consecutive reduction nodes on the chain and treat
        # bmm -> softmax -> bmm as ONE flash-attention-style kernel: a
        # generous allowance for what cross-op tiling could remove.
        REDUC = {"aten.sum", "aten.mean", "aten.var", "aten.amax", "aten.amin"}
        ncall = sum(hist.values())
        print(f"D aten_call_nodes={ncall}  barrier_nodes={sum(barrier_nodes.values())}  "
              f"CRITICAL_PATH_barriers={D}  critical_path_all_ops={Dall}", flush=True)
        # how wide is the graph at each barrier level - if width >> 1 the depth
        # term really is the only serial constraint
        width = collections.Counter(lvl_b[n] for n in g.nodes if n.op == "call_function")
        w = [width[i] for i in range(D + 1)]
        print(f"D level_widths(min/med/max)="
              f"{min(w)}/{sorted(w)[len(w)//2]}/{max(w)}  "
              f"levels={D+1}", flush=True)
        print("D barrier op histogram: " +
              ", ".join(f"{k.split('.')[-1]}={v}" for k, v in barrier_nodes.most_common()),
              flush=True)
        print("D top non-barrier ops: " +
              ", ".join(f"{k.split('.')[-1]}={v}" for k, v in hist.most_common(12)
                        if k not in BARRIER), flush=True)
        # the same depth, restricted to nodes on ONE longest path, printed so a
        # reader can audit the chain
        best = max((n for n in g.nodes if n.op == "call_function"),
                   key=lambda n: lvl_b[n])
        chain, cur = [], best
        while cur is not None:
            if cur.op == "call_function" and \
               ".".join(str(cur.target).split(".")[:2]) in BARRIER:
                chain.append(str(cur.target).split(".")[1])
            preds = [p for p in cur.all_input_nodes]
            cur = max(preds, key=lambda p: lvl_b.get(p, 0)) if preds else None
        chain.reverse()
        print(f"D longest-barrier-chain ({len(chain)}): {' -> '.join(chain)}", flush=True)


# -------------------------------------------------------------------- K
if "K" in S:
    from torch.profiler import profile, ProfilerActivity

    def group(name: str) -> str:
        n = name.lower()
        if any(k in n for k in ("gemm", "sgemm", "cutlass", "ampere_", "sm90", "sm80",
                                "gemv", "dot_kernel", "splitk")):
            return "GEMM (cuBLAS)"
        if "triton" in n:
            return "Triton"
        if "memcpy" in n or "memset" in n:
            return "memcpy/memset"
        return "other"

    torch._dynamo.reset()
    with torch.no_grad():
        f2 = FastDenoiser(m, L, device=dev, capture=False, manual_graph=False,
                          mode="default", preset=a.preset)
        f2.warmup(pair_cond=False)
        A_ = 5 + cfg.augment_size
        z = torch.zeros(L, dtype=torch.long, device=dev)
        pos = static["pos_noised"]
        d_ = dict(pos=pos, pos_noised=pos,
                  aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
                  seq=torch.full((L,), 20, dtype=torch.long, device=dev),
                  seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
                  residue_index=torch.arange(L, device=dev), chain_index=z,
                  batch_index=z, mask=torch.ones(L, dtype=torch.bool, device=dev),
                  t_pos=torch.full((L,), 0.5, device=dev),
                  t_seq=torch.ones(L, device=dev))
        p_ = {"pos": torch.zeros(L, A_, 3, device=dev),
              "local": torch.zeros(L, cfg.local_size, device=dev)}
        for need_aa in (False, True):
            for _ in range(5):
                f2(d_, p_, need_aa=need_aa)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(10):
                    f2(d_, p_, need_aa=need_aa)
                torch.cuda.synchronize()
            tot, cnt = collections.Counter(), collections.Counter()
            for e in prof.key_averages():
                if e.device_time_total <= 0 or e.device_type.name != "CUDA":
                    continue
                g_ = group(e.key)
                tot[g_] += e.device_time_total / 10 / 1000.0
                cnt[g_] += e.count // 10
            print(f"K preset={a.preset} need_aa={need_aa} "
                  f"kernels={sum(cnt.values())} gpu_busy_ms={sum(tot.values()):.4f}",
                  flush=True)
            for g_, ms in tot.most_common():
                print(f"K   {g_:<16} {cnt[g_]:>5} kernels {ms:7.4f} ms "
                      f"{ms/max(cnt[g_],1)*1000:6.2f} us each", flush=True)
            # per-kernel duration histogram against the MEASURED 1.166 us
            # per-launch floor (x_micro.py M1: marginal replay cost of an empty
            # kernel in a captured graph, serial chain). "excess" is the part
            # of GPU-busy that is NOT the floor.
            FLOOR = 1.166
            bins = [(0, 1.2), (1.2, 2), (2, 4), (4, 8), (8, 16), (16, 1e9)]
            hb, hms = collections.Counter(), collections.Counter()
            nk = 0.0
            excess = 0.0
            for e in prof.key_averages():
                if e.device_time_total <= 0 or e.device_type.name != "CUDA":
                    continue
                c = e.count // 10
                if c <= 0:
                    continue
                us = e.device_time_total / e.count
                nk += c
                excess += c * max(0.0, us - FLOOR)
                for lo, hi in bins:
                    if lo <= us < hi:
                        hb[(lo, hi)] += c
                        hms[(lo, hi)] += c * us / 1000.0
                        break
            print(f"K   HIST kernels={nk:.0f} floor_term={nk*FLOOR/1000:7.4f} ms "
                  f"above_floor={excess/1000:7.4f} ms", flush=True)
            for lo, hi in bins:
                if hb[(lo, hi)]:
                    print(f"K     {lo:>5.1f}-{hi if hi < 1e8 else 999:>5.0f} us  "
                          f"{hb[(lo,hi)]:>5} kernels  {hms[(lo,hi)]:7.4f} ms",
                          flush=True)
