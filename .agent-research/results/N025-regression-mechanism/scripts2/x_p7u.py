"""N025 - localise P7 to the PROLOGUE MODULE, compiled vs eager.

Four ways of computing the same six blocks' `pair_2` at one real mid-rollout
state, with the real DISTINCT per-block condition-neighbour index sets:

    seq_eager   six  blk.pair_2(...)            eager   <- the validated path
    seq_comp    six  blk.pair_2(...)            compiled
    pro_eager   one  cond_pair_stack(...)       eager
    pro_comp    one  cond_pair_stack(...)       compiled

`pro_eager - seq_eager` is P7's maths. `pro_comp - pro_eager` is inductor on the
batched shape. `seq_comp - seq_eager` is the same measurement for the shape P7
replaced, i.e. the control that says what "normal" compile drift looks like here.
"""
from __future__ import annotations
import argparse, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--state", required=True)
ap.add_argument("--device", default="cuda")
ap.add_argument("--config", default="default_vp")
ap.add_argument("--params", default="/hnvme/workspace/v181bb10-salad/default_vp-200k.npz")
a = ap.parse_args()

sys.path.insert(0, a.root)
from salad_torch import opt as salad_opt                                       # noqa
from salad_torch.config import get_config                                      # noqa
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz      # noqa
from salad_torch.modules.diffusion import (get_index_neighbours,               # noqa
    condition_neighbour_prologue, condition_neighbour_tail)
from salad_torch.modules.model import StructureDiffusionInference              # noqa

salad_opt.DET_SEGMENT = True
dev = torch.device(a.device)
cfg = get_config(a.config)
model = StructureDiffusionInference(cfg).eval().to(dev)
load_model(model.predict, load_param_npz(a.params), cfg, strict=True)
st = torch.load(a.state, map_location="cpu")
g = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in st.items()}
resi, chain, batch, mask = g["resi"], g["chain"], g["batch"], g["mask"]
prev_pos, draws = g["prev_pos"], g["uniform"]

stack = model.predict.diffusion.diffusion_stack
cps = stack.cond_pair_stack
assert cps is not None
print(f"[u] device={dev} tf32={torch.backends.cuda.matmul.allow_tf32} "
      f"cache in_sync={cps.in_sync(stack.blocks)} depth={cps.depth} "
      f"pair_size={cps.pair_size} prev_pos|max|={prev_pos.abs().max():.4f}", flush=True)

depth = len(stack.blocks)
with torch.no_grad():
    idx_nb = get_index_neighbours(16)(resi, chain, batch, mask)
    cd, cs = condition_neighbour_prologue(16, 16)(
        prev_pos, None, resi, chain, batch, mask, index_neighbours=idx_nb)
    nbrs = [condition_neighbour_tail(16, 0)(
                cd, cs, None, batch, mask,
                uniform=draws[1 + i * 3 + 1], uniform2=draws[1 + i * 3 + 2])
            for i in range(depth)]
    nb_stack = torch.stack(nbrs)
    ident = all(bool((nbrs[0] == nbrs[i]).all()) for i in range(depth))
    print(f"[u] per-block neighbour sets all identical? {ident}  K={nbrs[0].shape[-1]}",
          flush=True)

    seq_eager = [stack.blocks[i].pair_2(prev_pos, None, nbrs[i], resi, chain,
                                       batch, mask, cyclic_mask=None,
                                       init_pos=None)[0] for i in range(depth)]
    pro_eager = cps(prev_pos, None, nb_stack, resi, chain, batch, mask,
                    cyclic_mask=None, init_pos=None)[0]

    cps_c = torch.compile(cps.forward, mode="default", dynamic=False)
    p2_c = [torch.compile(stack.blocks[i].pair_2.forward, mode="default",
                          dynamic=False) for i in range(depth)]
    for _ in range(3):
        pro_comp = cps_c(prev_pos, None, nb_stack, resi, chain, batch, mask,
                         cyclic_mask=None, init_pos=None)[0]
        seq_comp = [p2_c[i](prev_pos, None, nbrs[i], resi, chain, batch, mask,
                            cyclic_mask=None, init_pos=None)[0]
                    for i in range(depth)]


def rep(label, fn):
    ds = [fn(i) for i in range(depth)]
    sc = max(seq_eager[i].abs().max().item() for i in range(depth)) or 1.0
    print(f"[u] {label:34} max|d|={max(ds):.6e}  rel={max(ds) / sc:.3e}  "
          f"per-block=[{' '.join(f'{d:.1e}' for d in ds)}]", flush=True)


rep("pro_eager vs seq_eager  (P7 maths)",
    lambda i: (pro_eager[i] - seq_eager[i]).abs().max().item())
rep("seq_comp  vs seq_eager  (CONTROL)",
    lambda i: (seq_comp[i] - seq_eager[i]).abs().max().item())
rep("pro_comp  vs pro_eager  (inductor/P7)",
    lambda i: (pro_comp[i] - pro_eager[i]).abs().max().item())
rep("pro_comp  vs seq_eager  (total)",
    lambda i: (pro_comp[i] - seq_eager[i]).abs().max().item())
rep("pro_comp  vs seq_comp",
    lambda i: (pro_comp[i] - seq_comp[i]).abs().max().item())
