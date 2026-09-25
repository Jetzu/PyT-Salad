"""Task 3: compare ONE Adam optimizer step (gradients + post-step parameter
delta) between torch and JAX, on the REAL training-step composition
(prepare_condition + Diffusion(predict_aa=False) + Diffusion.loss), same
released default_vp-200k weights, same batch, same noise, same neighbour
draws (both pinned via the fixed_uniform trick already used elsewhere).

Reference fixture: .agent-research/results/N004-porting-gaps/gen_gradstep_ref.py,
run inside the JAX container -> gradstep_default_vp.npz.
"""
import os
import sys

ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import torch

from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.losses.total import total_loss

torch.set_grad_enabled(True)

CONFIG_NAME = "default_vp"
PARAMS = f"/hnvme/workspace/v181bb10-salad/{CONFIG_NAME}-200k.npz"
FIX = os.path.join(HERE, f"gradstep_{CONFIG_NAME}.npz")

LR = 1e-3
B1, B2 = 0.9, 0.99
ADAM_EPS = 1e-9
CLIP = 0.1


def _t(arr):
    if arr.dtype == np.bool_:
        return torch.from_numpy(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return torch.from_numpy(arr.astype(np.int64))
    return torch.from_numpy(arr.astype(np.float32))


def main() -> int:
    for path in (PARAMS, FIX):
        if not os.path.exists(path):
            print(f"SKIP: missing {path}")
            return 0
    D = np.load(FIX)
    config = get_config(CONFIG_NAME)

    model = StructureDiffusionPredict(config)
    load_model(model, load_param_npz(PARAMS), config, strict=True)
    # snapshot BEFORE the step (independently of the checkpoint file, so the
    # delta comparison doesn't rely on re-reading disk twice)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    data = {k[3:]: _t(D[k]) for k in D.files if k.startswith("in_")}
    for k in ("pos_noised", "seq_noised", "t_pos", "t_seq"):
        data[k] = _t(D[f"noised_{k}"])
    n = data["mask"].shape[0]
    prev = {"pos": torch.zeros_like(data["pos"]),
           "local": torch.zeros(n, config.local_size)}
    uniform_model = _t(D["uniform_model"])
    uniform_loss = _t(D["uniform_loss"])

    result, _ = model(data, prev, predict_aa=False, uniform=uniform_model,
                      compute_distogram=True)
    total, terms = total_loss(config, data, result, fape_uniform=uniform_loss)

    print(f"torch total loss = {float(total):.10f}   "
         f"JAX total loss = {float(D['loss_total']):.10f}   "
         f"rel = {abs(float(total) - float(D['loss_total'])) / abs(float(D['loss_total'])):.3e}")
    print("  per-term:")
    for name in ("aa", "dssp", "pos", "pos_trajectory", "x", "rotation",
                "rotation_trajectory", "fape", "fape_trajectory", "local",
                "distogram", "violation"):
        ref = float(D[f"loss_{name}"])
        got = float(terms[name])
        rel = abs(got - ref) / max(abs(ref), 1e-12)
        print(f"    {name:22s} torch={got: .6f}  JAX={ref: .6f}  rel={rel:.3e}")

    model.zero_grad(set_to_none=True)
    total.backward()

    grad_norm_preclip = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1e30)   # measure only, no real clipping yet
    print(f"torch global grad norm (pre-clip) = {float(grad_norm_preclip):.6f}   "
         f"JAX = {float(D['global_grad_norm_preclip']):.6f}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP)

    # ---- 1. gradient comparison, name-mapped via the SAME converter table --
    grad_keys = [k for k in D.files if k.startswith("grad_")]
    grad_params = {k[len("grad_"):]: D[k] for k in grad_keys}
    dummy = StructureDiffusionPredict(config)
    ref_grad = load_model(dummy, grad_params, config, strict=True)

    print(f"\n{'== gradient comparison (torch vs JAX) ==':60s}")
    print(f"  {len(ref_grad)} parameters mapped from {len(grad_keys)} JAX leaves")
    worst = []
    n_none = 0
    n_zero_both = 0
    for name, p in model.named_parameters():
        ref = ref_grad[name].numpy()
        if p.grad is None:
            n_none += 1
            got = np.zeros_like(ref)
        else:
            got = p.grad.detach().numpy()
        denom = max(float(np.abs(ref).max()), 1e-12)
        rel = float(np.abs(got - ref).max()) / denom
        if np.abs(ref).max() == 0.0 and np.abs(got).max() == 0.0:
            n_zero_both += 1
        worst.append((rel, name, p.grad is None))
    worst.sort(reverse=True)
    print(f"  {n_none} params have grad=None on the torch side (documented "
         f"Haiku-allocated-but-unused .b biases)")
    print(f"  {n_zero_both} params are EXACTLY zero-gradient on BOTH sides")
    print("  worst 12 by peak-relative deviation:")
    for rel, name, is_none in worst[:12]:
        print(f"    {name:55s} rel={rel:.3e}{'  (torch grad=None)' if is_none else ''}")
    # overall verdict excluding the documented grad=None/zero-both cases
    real = [w for w in worst if not w[2]]
    print(f"  max rel over the {len(real)} params with a real torch gradient: "
         f"{real[0][0] if real else float('nan'):.3e}")

    # ---- 2. one optimizer step, then compare the parameter delta ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(B1, B2),
                                 eps=ADAM_EPS, weight_decay=0.0)
    optimizer.step()

    after_keys = [k for k in D.files if k.startswith("after_")]
    after_params = {k[len("after_"):]: D[k] for k in after_keys}
    dummy2 = StructureDiffusionPredict(config)
    ref_after = load_model(dummy2, after_params, config, strict=True)

    print(f"\n{'== post-step parameter delta comparison ==':60s}")
    worst2 = []
    for name, p in model.named_parameters():
        got_after = p.detach().numpy()
        got_delta = got_after - before[name].numpy()
        ref_delta = ref_after[name].numpy() - before[name].numpy()
        denom = max(float(np.abs(ref_delta).max()), 1e-12)
        rel = float(np.abs(got_delta - ref_delta).max()) / denom
        abs_err = float(np.abs(got_delta - ref_delta).max())
        worst2.append((rel, abs_err, name, float(np.abs(ref_delta).max())))
    worst2.sort(reverse=True)
    print("  worst 12 by peak-relative delta deviation "
         "(peak|ref_delta| shown for scale):")
    for rel, abs_err, name, peak in worst2[:12]:
        print(f"    {name:55s} rel={rel:.3e}  abs={abs_err:.3e}  "
             f"peak|ref_delta|={peak:.3e}")
    nontrivial = [w for w in worst2 if w[3] > 1e-12]
    print(f"  {len(worst2) - len(nontrivial)} / {len(worst2)} params have a "
         "numerically-zero JAX delta (excluded from the rel-error ranking above "
         "except where already shown)")
    print(f"  max rel over the {len(nontrivial)} params with a nonzero JAX "
         f"delta: {max(w[0] for w in nontrivial):.3e}" if nontrivial else "  n/a")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
