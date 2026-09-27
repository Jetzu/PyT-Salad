"""N030: compile ONE arm in a COLD process and dump all eleven outputs.

Used for the cross-process determinism control - N020 section 3d found that
`max_autotune_gemm`'s winner is a timing race, so three independent cold-cache
compiles of the SAME configuration disagreed by up to 8.87e-05 A. `base` is
run three times too, as the noise floor: N020 3c saw a no-op recompile differ
by 3.457e-05, and if that reproduces here nothing else in the table means
anything.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch._inductor.config as ind
import x_arm
from torch._functorch.compile_utils import fx_graph_cse

from x_common import build  # noqa: E402

OUT = os.environ["N030_DUMP"]
L = int(os.environ.get("N030_LEN", "100"))
ARM = os.environ.get("N030_ARM", "off")


def _cse(graph):
    graph.owning_module.graph = fx_graph_cse(graph)


ind.joint_custom_post_pass = _cse
ind.aggressive_fusion = True
from salad_torch import opt  # noqa: E402

assert opt.DET_SEGMENT, "run with SALAD_OPT_DETSEG=1"
x_arm.apply(ARM)
torch.set_float32_matmul_precision("highest")
torch.set_grad_enabled(False)
torch.manual_seed(0)
m, cfg, args = build(L, device="cuda")
core = torch.compile(m.diffusion.forward_core, dynamic=False)
out = core(*args, True, True, False)
torch.cuda.synchronize()
torch.save([o.detach().clone().cpu() if torch.is_tensor(o) else o for o in out], OUT)
print(f"wrote {OUT} L={L} ARM={ARM} max_autotune_gemm={ind.max_autotune_gemm}",
      flush=True)
