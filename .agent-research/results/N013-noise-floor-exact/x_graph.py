"""N013 - did P7 cost a Dynamo graph break? (notes/04 section 9's rule: check,
do not read.)  One COLD `torch._dynamo.explain` per variant."""
import os, sys
ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"; sys.path.insert(0, ROOT)
import torch
torch.set_float32_matmul_precision("highest")
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.modules.fast import FastDenoiser
from salad_torch import opt

L = int(sys.argv[1]) if len(sys.argv) > 1 else 100
cfg = get_config("default_vp")
dev = torch.device("cuda")
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"),
           cfg, strict=True)
for preset in ("exact", "deterministic"):
    fd = FastDenoiser(m, L, device=dev, preset=preset)
    fd._static_in = fd._blank_inputs()
    print(f"preset={preset} BATCH_COND={opt.BATCH_COND} DET_SEGMENT={opt.DET_SEGMENT} "
          f"aggressive_fusion={__import__('torch._inductor.config', fromlist=['x']).aggressive_fusion}")
    for need_aa in (False, True):
        for pair_cond in (False, True):
            torch._dynamo.reset()
            args = ([fd._static_in.get(k) for k in
                     __import__("salad_torch.modules.fast", fromlist=["x"])._CORE_ARGS]
                    + [fd.need_aux, need_aa, pair_cond])
            with torch.no_grad():
                ex = torch._dynamo.explain(m.diffusion.forward_core)(*args)
            print(f"  EXPLAIN {preset:<14} need_aa={need_aa!s:<5} pair_cond={pair_cond!s:<5} "
                  f"graphs {ex.graph_count} breaks {ex.graph_break_count} {ex.break_reasons}")
