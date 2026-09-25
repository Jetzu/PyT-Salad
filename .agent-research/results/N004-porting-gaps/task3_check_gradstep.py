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
    # NOTE: the real clip is applied further below, AFTER the gradient
    # comparison - `clip_grad_norm_` mutates `.grad` in place, and comparing
    # a CLIPPED torch gradient against JAX's unclipped `grads` dump would
    # scale every torch value down by the same ~1e-4 factor. Global-norm
    # clipping is a single uniform rescale, so it would not change cosine
    # similarity at all (scale-invariant), but it WOULD make the printed |g|
    # values confusingly tiny next to |r|, so the raw gradient is compared
    # here and clipping happens right before `optimizer.step()` instead.

    # ---- 1. gradient comparison, name-mapped via the SAME converter table --
    grad_keys = [k for k in D.files if k.startswith("grad_")]
    grad_params = {k[len("grad_"):]: D[k] for k in grad_keys}
    dummy = StructureDiffusionPredict(config)
    ref_grad = load_model(dummy, grad_params, config, strict=True)

    print(f"\n{'== gradient comparison (torch vs JAX) ==':60s}")
    print(f"  {len(ref_grad)} parameters mapped from {len(grad_keys)} JAX leaves")

    # Primary metric: COSINE similarity, not per-element peak-relative error.
    # Rationale (found empirically below): Adam's step-1 update is
    # m_hat/sqrt(v_hat) with m_hat=g, v_hat=g^2 (bias correction is exact at
    # t=1), i.e. update ~= sign(g)*lr PER ELEMENT, independent of |g|. A
    # parameter whose true gradient is near zero on a given element has that
    # element's SIGN set by ordinary fp32 noise (the same reduction-order
    # sensitivity already documented for the trunk/attention elsewhere in
    # this port) - Adam then turns that noise-level sign into a full +-lr
    # step. A peak-relative-error metric is dominated by exactly these
    # elements and says nothing about whether the gradient is actually
    # right; cosine similarity (whole-tensor and whole-model) is not.
    n_none = sum(1 for _, p in model.named_parameters() if p.grad is None)
    flat_g, flat_r = [], []
    noise_floor, cos_list = [], []
    for name, p in model.named_parameters():
        r = ref_grad[name].numpy()
        g = p.grad.detach().numpy() if p.grad is not None else np.zeros_like(r)
        flat_g.append(g.ravel()); flat_r.append(r.ravel())
        gn, rn = float(np.linalg.norm(g)), float(np.linalg.norm(r))
        if gn < 1e-4 and rn < 1e-4:
            noise_floor.append(name)
            continue
        cos = float(np.dot(g.ravel(), r.ravel()) / (gn * rn)) if gn and rn else float("nan")
        cos_list.append((cos, name, gn, rn))
    flat_g = np.concatenate(flat_g); flat_r = np.concatenate(flat_r)
    whole_cos = float(np.dot(flat_g, flat_r) /
                      (np.linalg.norm(flat_g) * np.linalg.norm(flat_r)))
    whole_rel = abs(float(np.linalg.norm(flat_g) - np.linalg.norm(flat_r))
                    ) / float(np.linalg.norm(flat_r))
    print(f"  {n_none} params have grad=None on torch (documented "
         f"Haiku-allocated-but-unused .b biases)")
    print(f"  {len(noise_floor)} params have BOTH |grad| < 1e-4 (below the "
         "noise floor, includes the grad=None ones)")
    print(f"  WHOLE-MODEL flattened gradient: cosine={whole_cos:.8f}  "
         f"norm torch={np.linalg.norm(flat_g):.4f} JAX={np.linalg.norm(flat_r):.4f} "
         f"rel={whole_rel:.3e}")
    cos_list.sort()
    n_good = sum(1 for c, *_ in cos_list if c > 0.999)
    n_ok = sum(1 for c, *_ in cos_list if 0.99 < c <= 0.999)
    n_bad = sum(1 for c, *_ in cos_list if c <= 0.99)
    print(f"  {len(cos_list)} params have non-negligible gradient; per-param "
         f"cosine: >0.999: {n_good}  0.99-0.999: {n_ok}  <=0.99: {n_bad}")
    print("  worst 8 by per-parameter cosine similarity:")
    for cos, name, gn, rn in cos_list[:8]:
        print(f"    {name:60s} cos={cos: .6f}  |g|={gn:.3e}  |r|={rn:.3e}")

    # ---- 2. NOW clip (see the note above `grad_norm_preclip`), then one
    #    optimizer step, then compare the parameter delta -------------------
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(B1, B2),
                                 eps=ADAM_EPS, weight_decay=0.0)
    optimizer.step()

    after_keys = [k for k in D.files if k.startswith("after_")]
    after_params = {k[len("after_"):]: D[k] for k in after_keys}
    dummy2 = StructureDiffusionPredict(config)
    ref_after = load_model(dummy2, after_params, config, strict=True)

    print(f"\n{'== post-step parameter delta comparison ==':60s}")
    print("  EXPECT per-element mismatches here even for a fully-correct optimizer:")
    print("  Adam's step-1 update is m_hat/sqrt(v_hat) with (exact bias correction")
    print("  at t=1) m_hat=g, v_hat=g^2, i.e. update ~= sign(g)*lr PER ELEMENT,")
    print("  independent of |g|. An element whose true gradient is ~0 has its sign")
    print("  set by ordinary fp32 noise, which Adam turns into a full +-lr step -")
    print("  a peak-relative metric over the whole tensor is dominated by exactly")
    print("  these elements. The gradient comparison above (cosine similarity) is")
    print("  the metric that actually answers 'does the optimizer see the right")
    print("  update direction'; this section is included to show WHY a raw")
    print("  post-step weight diff is the wrong thing to assert tightness on.")
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
