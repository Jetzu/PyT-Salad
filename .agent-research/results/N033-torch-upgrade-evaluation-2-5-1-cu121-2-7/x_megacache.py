"""N033 deliverable 5 - does Mega-Cache (torch.compiler.save/load_cache_artifacts)
actually help here?

Three phases, each a FRESH python process with its own EMPTY on-disk
TORCHINDUCTOR_CACHE_DIR (set by the caller via env var) so every phase's
`compile_time` is a genuine build, not served by inductor's normal on-disk FX
cache:

  cold  - baseline: compile with nothing preloaded. This is the true cold time.
  save  - compile (same as cold), then torch.compiler.save_cache_artifacts()
          the result to --cache-file.
  load  - BEFORE compiling, torch.compiler.load_cache_artifacts() the blob
          --cache-file (written by a prior `save` run in a DIFFERENT process
          with a DIFFERENT empty cache dir), then compile and see whether the
          artifact actually cuts compile_time.

Also dumps the output tensor (--out) so save vs load can be compared for
bitwise/numeric equality - a cache that changes the answer would be worse than
no cache at all.
"""
import argparse, os, sys, time, torch

ROOT = os.environ.get("SALAD_TREE", "/home/hpc/v181bb/v181bb10/SALAD_torch")
sys.path.insert(0, ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser
from salad_torch import opt

ap = argparse.ArgumentParser()
ap.add_argument("--phase", choices=["cold", "save", "load"], required=True)
ap.add_argument("--length", type=int, default=100)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--preset", default="fast")
ap.add_argument("--cache-file", required=True)
ap.add_argument("--out", default=None)
a = ap.parse_args()


def _tensors(x):
    if torch.is_tensor(x):
        return [x]
    if isinstance(x, dict):
        out = []
        for v in x.values():
            out += _tensors(v)
        return out
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out += _tensors(v)
        return out
    return []


if a.phase == "load":
    # torch 2.7.1 footgun (found the hard way, job 908347): calling
    # load_cache_artifacts() as the very FIRST thing to touch torch in a
    # fresh process crashes with "cannot import name 'CacheArtifactManager'
    # from partially initialized module 'torch.compiler._cache' (circular
    # import)". torch.compiler.__init__ imports ._cache, whose own import
    # chain (via _inductor.remote_cache -> _dynamo.utils -> ... -> _dynamo.pgo)
    # re-imports torch.compiler._cache while it is still mid-import. Touching
    # torch._dynamo ONCE, top-down, before calling load_cache_artifacts avoids
    # ever re-entering it from the middle.
    import torch._dynamo  # noqa: F401  (load-order fix, see comment above)
    blob = open(a.cache_file, "rb").read()
    t0 = time.perf_counter()
    torch.compiler.load_cache_artifacts(blob)
    t_load = time.perf_counter() - t0
    print(f"MEGACACHE load_cache_artifacts bytes={len(blob)} wall={t_load:.3f}s", flush=True)

torch.manual_seed(0)  # else FastDenoiser's own internal (caller-unseeable)
# neighbour/uniform draws differ between processes and confound the
# save-vs-load output comparison with ordinary cross-process randomness that
# has nothing to do with the cache (missed in the first run, job 908347).
torch.set_float32_matmul_precision("highest")
dev = torch.device("cuda")
cfg = get_config(a.config)
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz(f"/hnvme/workspace/v181bb10-salad/{a.config}-200k.npz"), cfg, strict=True)

L = a.length
A = 5 + cfg.augment_size
g = torch.Generator(device=dev).manual_seed(0)
z = torch.zeros(L, dtype=torch.long, device=dev)
pos = torch.randn(L, A, 3, generator=g, device=dev) * 4
d = dict(pos=pos, pos_noised=pos, aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
         seq=torch.full((L,), 20, dtype=torch.long, device=dev),
         seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
         residue_index=torch.arange(L, device=dev), chain_index=z, batch_index=z,
         mask=torch.ones(L, dtype=torch.bool, device=dev),
         cyclic_mask=torch.zeros(L, dtype=torch.bool, device=dev),
         t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev))
p = {"pos": torch.zeros(L, A, 3, device=dev), "local": torch.zeros(L, cfg.local_size, device=dev)}

t0 = time.perf_counter()
with torch.no_grad():
    fast = FastDenoiser(m, L, device=dev, preset=a.preset).warmup(
        pair_cond=not opt.SKIP_PAIR_COND)
torch.cuda.synchronize()
t_compile = time.perf_counter() - t0

with torch.no_grad():
    out = fast(d, p)
torch.cuda.synchronize()

print(f"MEGACACHE RESULT phase={a.phase:<5} preset={a.preset} L={L} "
      f"torch={torch.__version__} compile_time={t_compile:.2f}s", flush=True)

if a.phase == "save":
    blob, info = torch.compiler.save_cache_artifacts()
    with open(a.cache_file, "wb") as f:
        f.write(blob)
    print(f"MEGACACHE saved bytes={len(blob)} info={info}", flush=True)

if a.out:
    torch.save([t.detach().cpu() for t in _tensors(out)], a.out)
    print(f"MEGACACHE wrote {len(_tensors(out))} output tensors to {a.out}", flush=True)
