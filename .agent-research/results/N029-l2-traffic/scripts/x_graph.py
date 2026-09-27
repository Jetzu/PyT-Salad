"""N029/A - byte attribution off the traced ATEN graph of one lean forward.

Prints, per source-site and per module:
  * UNFUSED traffic   (every node's inputs+outputs, i.e. no-fusion upper bound)
  * MIN traffic       (barrier-op inputs+outputs, each tensor once = N016's 926.9 MiB)
and separately:
  * every matmul's shape, its operand bytes, and a tile-amplification estimate
  * structurally duplicate subexpressions still left after inductor's CSE
"""
import collections, os, sys, hashlib
import torch

ROOT = os.environ.get("SALAD_ROOT", "/hnvme/workspace/v181bb10-salad/n029-l2")
sys.path.insert(0, ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.fast import FastDenoiser, PRESETS
from salad_torch.modules.model import StructureDiffusionPredict

torch.set_float32_matmul_precision("highest")
dev = torch.device("cuda")
cfg = get_config("default_vp")
L = int(os.environ.get("SALAD_L", "100"))
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)
A = 5 + cfg.augment_size
z = torch.zeros(L, dtype=torch.long, device=dev)
pos = torch.randn(L, A, 3, device=dev) * 4
core = m.diffusion.forward_core
g = torch.Generator(device="cpu").manual_seed(0)
uni = torch.rand(1 + 6 * 3, L, L, generator=g).to(dev)
ARGS = dict(
    pos_noised=pos, prev_pos=torch.zeros(L, A, 3, device=dev),
    prev_local=torch.zeros(L, cfg.local_size, device=dev),
    condition=torch.zeros(L, cfg.local_size, device=dev),
    dmap=torch.zeros(L, L, device=dev), omap=torch.zeros(L, L, 3, device=dev),
    dmap_mask=torch.zeros(L, L, dtype=torch.bool, device=dev),
    flags=torch.zeros(L, L, 3, device=dev),
    t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev),
    seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
    aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
    pos_gt=pos, resi=torch.arange(L, device=dev), chain=z, batch=z,
    mask=torch.ones(L, dtype=torch.bool, device=dev),
    cyclic_mask=None, uniform=uni,
    need_aux=False, need_aa=False, pair_cond=False,
)

BARRIER = {
    "aten.mm", "aten.bmm", "aten.addmm", "aten.baddbmm", "aten.matmul",
    "aten._softmax", "aten._log_softmax", "aten.softmax",
    "aten.native_layer_norm", "aten.layer_norm",
    "aten.sum", "aten.mean", "aten.amax", "aten.amin", "aten.max", "aten.min",
    "aten.var", "aten.var_mean", "aten.std", "aten.prod", "aten.cumsum",
    "aten.linalg_vector_norm", "aten.norm",
    "aten.sort", "aten.argsort", "aten.topk", "aten.scatter_add",
    "aten.index_put", "aten.index_put_", "aten.scatter", "aten.scatter_reduce",
    "aten._unique", "aten.nonzero", "aten.unique_consecutive",
}
VIEW = ("view", "permute", "unsqueeze", "squeeze", "slice", "expand", "reshape",
        "t.default", "transpose", "detach", "select", "as_strided",
        "_unsafe_view", "alias", "lift_fresh")

from torch._dynamo.backends.common import aot_autograd
captured = []
def fw(gm, ex):
    captured.append(gm); return gm.forward
torch._dynamo.reset()
fn = torch.compile(core, backend=aot_autograd(fw_compiler=fw), dynamic=False)
with torch.no_grad():
    try:
        fn(**ARGS)
    except Exception as e:
        print("note: raw fx exec raised", type(e).__name__, str(e)[:100], flush=True)
gm = captured[0]
print(f"graph nodes={len(list(gm.graph.nodes))}", flush=True)

def nbytes(x):
    v = x.meta.get("val", None) if hasattr(x, "meta") else None
    if v is None or not hasattr(x, "meta"):
        return 0
    try:
        return int(v.numel()) * v.element_size()
    except Exception:
        return 0

def site(n):
    st = n.meta.get("stack_trace", "") or ""
    best = None
    for line in st.splitlines():
        line = line.strip()
        if line.startswith("File \"") and "salad_torch" in line:
            f = line.split('"')[1].split("salad_torch/")[-1]
            ln = line.split("line ")[1].split(",")[0]
            fnm = line.split(" in ")[-1] if " in " in line else "?"
            best = f"{f}:{ln} {fnm}"
    return best or "<no-stack>"

def mod(n):
    ms = n.meta.get("nn_module_stack", None)
    if not ms:
        return "<none>"
    keys = list(ms.values())
    return keys[-1][0] if keys else "<none>"

unf = collections.Counter()      # site -> unfused bytes
minb = collections.Counter()     # site -> min-materialised bytes
unf_m = collections.Counter()
minb_m = collections.Counter()
seen = set()
tot_unf = tot_min = 0
opcnt = collections.Counter()
for n in gm.graph.nodes:
    if n.op != "call_function":
        continue
    nm = str(n.target)
    short = ".".join(nm.split(".")[:2])
    opcnt[short] += 1
    s, mo = site(n), mod(n)
    if not any(v in nm for v in VIEW):
        b = nbytes(n) + sum(nbytes(x) for x in n.all_input_nodes)
        unf[s] += b; unf_m[mo] += b; tot_unf += b
    if short in BARRIER:
        for x in list(n.all_input_nodes) + [n]:
            if id(x) in seen:
                continue
            seen.add(id(x))
            bb = nbytes(x)
            minb[s] += bb; minb_m[mo] += bb; tot_min += bb

print(f"\nTOTAL unfused(non-view in+out) = {tot_unf/2**20:.1f} MiB"
      f"   min-materialised = {tot_min/2**20:.1f} MiB", flush=True)

print("\n== UNFUSED traffic by source site (top 30) ==", flush=True)
for s, b in unf.most_common(30):
    print(f"{b/2**20:9.1f} MiB {100*b/tot_unf:5.1f}%  min={minb[s]/2**20:8.1f}  {s}", flush=True)

print("\n== MIN-MATERIALISED traffic by source site (top 25) ==", flush=True)
for s, b in minb.most_common(25):
    print(f"{b/2**20:9.1f} MiB {100*b/tot_min:5.1f}%  {s}", flush=True)

print("\n== by nn_module (top 20, unfused) ==", flush=True)
for s, b in unf_m.most_common(20):
    print(f"{b/2**20:9.1f} MiB  min={minb_m[s]/2**20:8.1f}  {s}", flush=True)

# ---------------- matmuls ----------------
print("\n== MATMULS: shape, ideal operand bytes, tile-amplified L2 estimate ==", flush=True)
def shapes(n):
    out = []
    for x in n.all_input_nodes:
        v = x.meta.get("val", None)
        if v is not None and hasattr(v, "shape"):
            out.append(tuple(int(d) for d in v.shape))
    return out
mm_tab = collections.Counter()
mm_ideal = collections.Counter()
mm_amp = collections.Counter()
for n in gm.graph.nodes:
    if n.op != "call_function":
        continue
    nm = str(n.target)
    if not nm.startswith(("aten.mm", "aten.bmm", "aten.addmm", "aten.baddbmm")):
        continue
    sh = [s for s in shapes(n) if len(s) in (2, 3)]
    if len(sh) < 2:
        continue
    a, b = sh[-2], sh[-1]
    if len(a) == 3:
        B, M, K = a; N = b[-1]
    else:
        B = 1; M, K = a; N = b[-1]
    key = (B, M, K, N)
    ideal = 4 * B * (M * K + K * N + M * N)
    # cuBLAS fp32 SIMT tiles are 32x32 or 64x64; assume 64x64 when both >=64
    BM = 64 if M >= 64 else 32
    BN = 64 if N >= 64 else 32
    import math
    amp = 4 * B * (M * K * math.ceil(N / BN) + K * N * math.ceil(M / BM) + M * N)
    mm_tab[key] += 1; mm_ideal[key] += ideal; mm_amp[key] += amp
tot_i = sum(mm_ideal.values()); tot_a = sum(mm_amp.values())
print(f"{'B':>3} {'M':>6} {'K':>5} {'N':>5} {'cnt':>4} {'idealMiB':>9} {'ampMiB':>8}", flush=True)
for k, c in sorted(mm_tab.items(), key=lambda x: -mm_amp[x[0]]):
    print(f"{k[0]:3d} {k[1]:6d} {k[2]:5d} {k[3]:5d} {c:4d} "
          f"{mm_ideal[k]/2**20:9.2f} {mm_amp[k]/2**20:8.2f}", flush=True)
print(f"MATMUL TOTAL: ideal {tot_i/2**20:.1f} MiB, tile-amplified {tot_a/2**20:.1f} MiB, "
      f"count {sum(mm_tab.values())}", flush=True)

# ---------------- duplicate subexpressions ----------------
print("\n== structurally duplicate subexpressions (post-dispatch, pre-inductor-CSE) ==", flush=True)
h = {}
def key_of(n):
    if n in h:
        return h[n]
    if n.op == "placeholder":
        k = "P:" + n.name
    elif n.op == "get_attr":
        k = "G:" + str(n.target)
    elif n.op == "call_function":
        parts = [str(n.target)]
        def enc(a):
            if isinstance(a, torch.fx.Node):
                return key_of(a)
            if isinstance(a, (list, tuple)):
                return "[" + ",".join(enc(x) for x in a) + "]"
            return repr(a)
        parts += [enc(a) for a in n.args]
        parts += [f"{kk}={enc(vv)}" for kk, vv in sorted(n.kwargs.items())]
        k = hashlib.blake2b("|".join(parts).encode(), digest_size=12).hexdigest()
    else:
        k = "O:" + n.name
    h[n] = k
    return k
dup = collections.defaultdict(list)
for n in gm.graph.nodes:
    if n.op == "call_function":
        dup[key_of(n)].append(n)
groups = [(v, sum(nbytes(x) for x in v[1:])) for v in dup.values() if len(v) > 1]
groups.sort(key=lambda x: -x[1])
tot_dup_b = sum(b for _, b in groups)
tot_dup_n = sum(len(v) - 1 for v, _ in groups)
print(f"duplicate groups={len(groups)}  redundant nodes={tot_dup_n}  "
      f"their output bytes={tot_dup_b/2**20:.2f} MiB", flush=True)
for v, b in groups[:25]:
    print(f"  x{len(v):3d} {b/2**20:8.3f} MiB  {str(v[0].target).split('.')[1]:<22} {site(v[0])}", flush=True)
print("N029-GRAPH-DONE", flush=True)
