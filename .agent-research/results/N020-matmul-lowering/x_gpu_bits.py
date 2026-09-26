"""N020 control #2: the same bitwise comparison ON THE H100.

The CPU control proves the SOURCE change (einsum -> mm) is exact, but CPU has no
Triton matmul templates at all, so it cannot say anything about which kernel
inductor picks on the GPU. Under `SALAD_OPT_DETSEG=1` the only source of CUDA
run-to-run nondeterminism in this model (the segment reduction's atomicAdd) is
gone, so a GPU comparison is bitwise too - the same control N013 used for
`aggressive_fusion` and N019 for design grouping.

One arm per process; the sbatch compares the dumps afterwards.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch._inductor.config as ind

OUT = os.environ["N020_DUMP"]
L = int(os.environ.get("N020_LEN", "100"))
if os.environ.get("N020_CSE", "1") == "1":
    from torch._functorch.compile_utils import fx_graph_cse

    def _cse(graph):
        graph.owning_module.graph = fx_graph_cse(graph)

    ind.joint_custom_post_pass = _cse
ind.aggressive_fusion = os.environ.get("N020_AGGR", "1") == "1"

from x_common import build                                        # noqa: E402
from salad_torch import opt                                       # noqa: E402

assert opt.DET_SEGMENT, "run with SALAD_OPT_DETSEG=1 - this is the bitwise control"
ind.max_autotune_gemm = opt.GEMM_TEMPLATES
if ind.max_autotune_gemm:
    ind.max_autotune_gemm_backends = "ATEN,TRITON"
torch.set_float32_matmul_precision("highest")
torch.set_grad_enabled(False)
torch.manual_seed(0)
m, cfg, args = build(L, device="cuda")
core = torch.compile(m.diffusion.forward_core, dynamic=False)
out = core(*args, True, True, False)
torch.cuda.synchronize()
torch.save([o.detach().clone().cpu() if torch.is_tensor(o) else o for o in out], OUT)
print(f"wrote {OUT} L={L} MM_MODE={opt.MM_MODE} TPL={opt.GEMM_TEMPLATES} "
      f"DETSEG={opt.DET_SEGMENT}", flush=True)
