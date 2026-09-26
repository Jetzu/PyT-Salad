"""N016/R2 - actual DRAM traffic of one lean forward, under Nsight Compute."""
import sys, torch
ROOT = "/hnvme/workspace/v181bb10-salad/n016-floor"; sys.path.insert(0, ROOT)
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.fast import FastDenoiser
from salad_torch.modules.model import StructureDiffusionPredict
torch.set_float32_matmul_precision("highest")
dev = torch.device("cuda"); cfg = get_config("default_vp"); L = 100
m = StructureDiffusionPredict(cfg).eval().to(dev)
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)
A = 5 + cfg.augment_size
z = torch.zeros(L, dtype=torch.long, device=dev); pos = torch.randn(L, A, 3, device=dev) * 4
d = dict(pos=pos, pos_noised=pos, aa_gt=torch.full((L,), 20, dtype=torch.long, device=dev),
         seq=torch.full((L,), 20, dtype=torch.long, device=dev),
         seq_noised=torch.full((L,), 20, dtype=torch.long, device=dev),
         residue_index=torch.arange(L, device=dev), chain_index=z, batch_index=z,
         mask=torch.ones(L, dtype=torch.bool, device=dev),
         t_pos=torch.full((L,), 0.5, device=dev), t_seq=torch.ones(L, device=dev))
p = {"pos": torch.zeros(L, A, 3, device=dev), "local": torch.zeros(L, cfg.local_size, device=dev)}
with torch.no_grad():
    f = FastDenoiser(m, L, device=dev, capture=False, manual_graph=False, mode="default")
    f.warmup(pair_cond=False)
    for _ in range(5):
        f(d, p, need_aa=False)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("measure")
    f(d, p, need_aa=False)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print("NCU-SCRIPT-DONE", flush=True)
