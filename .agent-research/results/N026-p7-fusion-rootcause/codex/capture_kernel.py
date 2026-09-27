"""N025 - localise the P7 miscompile: which block, and which inductor pass.

`trajectory` is `torch.stack([pos after each block])`, so the first block whose
entry diverges is where the compiled P7 path first goes wrong. Probe arms turn
individual inductor passes off to see which one owns the 9.59 A.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--state", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dump", action="store_true")
ap.add_argument("--compiled", action="store_true")
ap.add_argument("--probe", default="")   # aggr0 | nopat | inline | reduceov | freeze0
ap.add_argument("--p7", choices=("on", "off"), required=True)
ap.add_argument("--fusion-limit", type=int, default=-1)
ap.add_argument("--veto-pair", nargs=2, default=None)
ap.add_argument("--capture", required=True)
ap.add_argument("--num_aa", type=int, default=100)
ap.add_argument("--warm_steps", type=int, default=30)
ap.add_argument("--num_steps", type=int, default=250)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--config", default="default_vp")
ap.add_argument("--params", default="/hnvme/workspace/v181bb10-salad/default_vp-200k.npz")
a = ap.parse_args()

sys.path.insert(0, a.root)
import torch._dynamo.config as _dyn
import torch._inductor.config as _ind
from salad_torch import opt as salad_opt                                       # noqa
from salad_torch.config import get_config                                      # noqa
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz      # noqa
from salad_torch.diffusion.sampling import make_initial_data                   # noqa
from salad_torch.diffusion.schedules import parse_timescale                    # noqa
from salad_torch.modules.diffusion import uniform_draw_count                   # noqa
from salad_torch.modules.model import StructureDiffusionInference, apply_diffusion  # noqa

salad_opt.DET_SEGMENT = True
salad_opt.BATCH_COND = a.p7 == "on"
_ind.trace.enabled = bool(int(os.environ.get("TORCH_COMPILE_DEBUG", "0")))
_ind.trace.debug_dir = os.environ.get("N026_TRACE_DIR")
mode = "default"
if a.probe == "aggr0":
    _ind.aggressive_fusion = False
elif a.probe == "nopat":
    _ind.pattern_matcher = False
elif a.probe == "inline":
    _dyn.inline_inbuilt_nn_modules = True
elif a.probe == "noepi":
    _ind.epilogue_fusion = False
elif a.probe == "reduceov":
    mode = "reduce-overhead"
elif a.probe == "nofuse":
    _ind.max_fusion_size = 1
elif a.probe.startswith("fuse"):
    _ind.max_fusion_size = int(a.probe[4:])
elif a.probe == "nosplitred":
    _ind.split_reductions = False
elif a.probe == "nopersred":
    _ind.triton.persistent_reductions = False
elif a.probe == "nomultikern":
    _ind.triton.multi_kernel = 0
elif a.probe == "cooploose":
    _ind.triton.cooperative_reductions = False
print(f"[b] probe={a.probe or 'none'} mode={mode} BATCH_COND="
      f"{getattr(salad_opt, 'BATCH_COND', 'absent')} compiled={a.compiled} "
      f"aggr={_ind.aggressive_fusion} patmatch={_ind.pattern_matcher} "
      f"inline={_dyn.inline_inbuilt_nn_modules}", flush=True)

# Experimental node-granular fusion gate. Counts only scheduler-approved fusions;
# the first --fusion-limit are permitted, later pairs rejected. Each invocation
# is a fresh process so Inductor compiles a distinct graph.
if a.fusion_limit >= 0 or a.veto_pair:
    from torch._inductor.scheduler import Scheduler
    _speedup = Scheduler.speedup_by_fusion
    _counter = [0]
    def _limited(self, node1, node2):
        pair = (node1.get_name(), node2.get_name())
        permit = (a.fusion_limit < 0 or _counter[0] < a.fusion_limit)
        if a.veto_pair and pair == tuple(a.veto_pair):
            permit = False
        print(f"[fusion] {_counter[0]} {int(permit)} {pair[0]} + {pair[1]}", flush=True)
        _counter[0] += 1
        return permit and _speedup(self, node1, node2)
    Scheduler.speedup_by_fusion = _limited

# Hook the exact generated Triton call when PyCodeCache loads the wrapper.
from torch._inductor.codecache import PyCodeCache
_orig_load = PyCodeCache.load_by_key_path.__func__
_capture_done = [False]
@classmethod
def _capture_load(cls, key, path, linemap=None, attrs=None):
    mod = _orig_load(cls, key, path, linemap, attrs)
    for name, kernel in list(vars(mod).items()):
        if not (name.startswith("triton_") and "repeat_stack_sub_where" in name):
            continue
        if getattr(kernel, "_n026_hooked", False):
            continue
        original_run = kernel.run
        def wrapped(*args, grid, stream, _run=original_run, _name=name, **kwargs):
            inputs = [x.detach().cpu().clone() if torch.is_tensor(x) else x for x in args]
            result = _run(*args, grid=grid, stream=stream, **kwargs)
            if not _capture_done[0]:
                torch.cuda.synchronize()
                outs = [args[-3].detach().cpu().clone(), args[-2].detach().cpu().clone()]
                torch.save(dict(kernel=_name, args=inputs, outputs=outs), a.capture)
                _capture_done[0] = True
                print("[capture] saved", a.capture, _name, "numargs", len(args), flush=True)
            return result
        kernel.run = wrapped
        kernel._n026_hooked = True
    return mod
PyCodeCache.load_by_key_path = _capture_load

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda")
cfg = get_config(a.config)
model = StructureDiffusionInference(cfg).eval().to(dev)
load_model(model.predict, load_param_npz(a.params), cfg, strict=True)
L, nd = a.num_aa, uniform_draw_count(cfg)
ORDER = ("pos_noised", "prev_pos", "prev_local", "condition", "dmap", "omap",
         "dmap_mask", "flags", "t_pos", "t_seq", "seq_noised", "aa_gt",
         "pos_gt", "resi", "chain", "batch", "mask", "cyclic_mask", "uniform")

if a.dump:
    resi = torch.arange(L, device=dev)
    chain = torch.zeros(L, dtype=torch.long, device=dev)
    data, init_prev = make_initial_data(L, resi, chain, None, device=dev)
    prev = init_prev
    ts = parse_timescale("cosine(t)")
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    cond = model.predict.prepare_condition(data); pc = cond["pair_condition"]
    salad_opt.BATCH_COND = False
    for step in range(a.warm_steps + 1):
        raw_t = 1 - step / a.num_steps
        data["t_pos"] = ts(raw_t) * torch.ones_like(data["t_pos"])
        data["t_seq"] = torch.ones_like(data["t_seq"])
        if raw_t < 0.8:
            prev = init_prev
        with torch.no_grad():
            noised = apply_diffusion(cfg, data, generator=gen)
        d = dict(data, **noised)
        uni = torch.rand(nd, L, L, device=dev, generator=gen).clamp_(1e-6, 1 - 1e-6)
        pack = dict(pos_noised=d["pos_noised"], prev_pos=prev["pos"],
                    prev_local=prev["local"], condition=cond["condition"],
                    dmap=pc["dmap"], omap=pc["omap"], dmap_mask=pc["dmap_mask"],
                    flags=pc["flags"], t_pos=d["t_pos"], t_seq=d["t_seq"],
                    seq_noised=d["seq_noised"], aa_gt=d["aa_gt"], pos_gt=d["pos"],
                    resi=d["residue_index"], chain=d["chain_index"],
                    batch=d["batch_index"], mask=d["mask"], cyclic_mask=None,
                    uniform=uni)
        if step == a.warm_steps:
            torch.save({k: (v.cpu() if torch.is_tensor(v) else v)
                        for k, v in pack.items()}, a.state)
            print(f"[b] dumped step {a.warm_steps}, prev_pos|max|="
                  f"{pack['prev_pos'].abs().max():.3f}", flush=True)
            raise SystemExit(0)
        with torch.no_grad():
            out = model.predict.diffusion.forward_core(
                *[pack[k] for k in ORDER], True, True, False)
        data["pos"] = out[0]; data["seq"] = out[3].argmax(-1)
        prev = {"pos": out[0], "local": out[1]}

st = torch.load(a.state, map_location="cpu")
args = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in st.items()}
fn = model.predict.diffusion.forward_core
if a.compiled:
    fn = torch.compile(fn, mode=mode, dynamic=False)
argl = [args[k] for k in ORDER] + [True, True, False]
with torch.no_grad():
    for _ in range(1):
        out = fn(*argl)
np.savez(a.out, pos=out[0].detach().cpu().numpy(),
         local=out[1].detach().cpu().numpy(),
         trajectory=out[2].detach().cpu().numpy())
print(f"[b] wrote {a.out} traj shape {tuple(out[2].shape)}", flush=True)
