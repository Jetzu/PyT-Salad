"""Run scripts/generate.py with optional in-process vetoes, no checkout needed.

SALAD_NOCSE=1  -> replace `fx_graph_cse` with the IDENTITY. FastDenoiser still
                  installs a `joint_custom_post_pass`, so inductor's FX cache is
                  still disabled (N022) and the pass still runs - it just
                  transforms nothing. Isolates the CSE TRANSFORM.
                  NB: do NOT `del` the attribute - torch._functorch.partitioners
                  does `from .compile_utils import fx_graph_cse` at module import
                  time and an absent attribute breaks that import outright
                  (this is what killed job 904697's arm D).
SALAD_NOCSE=2  -> let FastDenoiser install it, then set
                  torch._inductor.config.joint_custom_post_pass = None just
                  before warmup() (torch.compile is lazy: nothing is compiled
                  until the first _build call inside warmup). Full veto.
"""
import atexit
import os
import runpy
import sys

sys.argv = sys.argv[1:]
_root = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
sys.path.insert(0, _root)

_mode = os.environ.get("SALAD_NOCSE", "")

if _mode == "1":
    import torch._functorch.compile_utils as cu
    _orig = cu.fx_graph_cse
    cu.fx_graph_cse = lambda g: g
    print("[wrap] fx_graph_cse -> identity (CSE transform vetoed)", flush=True)
elif _mode == "2":
    import torch._inductor.config as _ic
    import salad_torch.modules.fast as _f

    _real_warmup = _f.FastDenoiser.warmup

    def _warmup(self, *a, **kw):
        _ic.joint_custom_post_pass = None
        print("[wrap] joint_custom_post_pass reset to None before compile",
              flush=True)
        return _real_warmup(self, *a, **kw)

    _f.FastDenoiser.warmup = _warmup
    print("[wrap] FastDenoiser.warmup patched (full CSE veto)", flush=True)


def _report():
    try:
        import torch
        import torch._inductor.config as ic
        print(f"[wrap] joint_custom_post_pass={ic.joint_custom_post_pass!r} "
              f"aggressive_fusion={ic.aggressive_fusion} "
              f"matmul_precision={torch.get_float32_matmul_precision()}", flush=True)
        from salad_torch import opt as o
        print(f"[wrap] BATCH_COND={o.BATCH_COND} DET_SEGMENT={o.DET_SEGMENT}",
              flush=True)
    except Exception as e:
        print(f"[wrap] report failed: {e}", flush=True)


atexit.register(_report)
runpy.run_path(sys.argv[0], run_name="__main__")
