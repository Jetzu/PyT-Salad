"""N020 control: is the mm/addmm lowering BIT-EXACT against the einsum form?

CPU is deterministic by construction, so this is a bitwise answer and not a
tolerance - which is the point: the Angstrom parity floor is NOT sufficient for
a change to how every matmul in the model lowers (N017 caught
TORCHINDUCTOR_FREEZING=1 at pos 8.8e-06, under the 2e-05 floor and not exact).

One arm per process (SALAD_OPT_MM is read at import time); each writes a dump.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ["SALAD_TREE"])
import torch
import torch._inductor.config as ind

OUT = os.environ["N020_DUMP"]
L = int(os.environ.get("N020_LEN", "64"))
# match the shipped compile as closely as CPU allows
if os.environ.get("N020_CSE", "1") == "1":
    from torch._functorch.compile_utils import fx_graph_cse

    def _cse(graph):
        graph.owning_module.graph = fx_graph_cse(graph)

    ind.joint_custom_post_pass = _cse
ind.aggressive_fusion = os.environ.get("N020_AGGR", "1") == "1"
# N020 deliverable-1 gap: does GEMM_TEMPLATES do anything on CPU at all? Wired
# the same way N020_AGGR is (a direct inductor config write in this process,
# not via FastDenoiser), read from opt.GEMM_TEMPLATES so SALAD_OPT_GEMMTPL=0|1
# in the environment controls it exactly like on GPU (x_gpu_bits.py).
from salad_torch import opt                                       # noqa: E402
ind.max_autotune_gemm = opt.GEMM_TEMPLATES
if ind.max_autotune_gemm:
    ind.max_autotune_gemm_backends = "ATEN,TRITON"

from x_common import build                                        # noqa: E402

torch.set_grad_enabled(False)
torch.manual_seed(0)
m, cfg, args = build(L)
if os.environ.get("N020_EAGER") == "1":
    core = m.diffusion.forward_core
else:
    core = torch.compile(m.diffusion.forward_core, dynamic=False)
out = core(*args, True, True, False)
torch.save([o.detach().clone() if torch.is_tensor(o) else o for o in out], OUT)
print(f"wrote {OUT} L={L} MM_MODE={opt.MM_MODE} TPL={opt.GEMM_TEMPLATES} "
      f"tensors={len(out)}", flush=True)
