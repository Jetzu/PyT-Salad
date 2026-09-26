"""N013/P7 - how big is the batched prologue's perturbation, on the DETERMINISTIC
backend, across a real trajectory?

One forward at a time: the sampler is advanced with P7 OFF, and at every step
the same state is run once more with P7 ON. That isolates the per-step
reassociation from the sampler's chaos amplification. CPU, so the number
printed is the whole effect.
"""
import sys, os
ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, torch                                          # noqa: E402
torch.set_grad_enabled(False); torch.set_float32_matmul_precision("highest")
import tests.test_sampling_parity as T                             # noqa: E402
from salad_torch import opt                                        # noqa: E402
from salad_torch.config import get_config                          # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.modules.model import StructureDiffusionPredict    # noqa: E402

NSTEP = int(sys.argv[1]) if len(sys.argv) > 1 else 24
C = get_config("default_vp")
D = np.load("fixtures/sampling_default_vp.npz")
L, S = int(D["in_length"]), int(D["in_num_steps"])
A = 5 + C.augment_size
m = StructureDiffusionPredict(C).eval()
load_model(m, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"),
           C, strict=True)
noise, uni = torch.from_numpy(D["noise"]), torch.from_numpy(D["uniform"])
z = torch.zeros(L, dtype=torch.long)
data = dict(pos=torch.zeros(L, A, 3), aa_gt=torch.full((L,), 20, dtype=torch.long),
            seq=torch.full((L,), 20, dtype=torch.long), residue_index=torch.arange(L),
            chain_index=z, batch_index=z, mask=torch.ones(L, dtype=torch.bool),
            cyclic_mask=torch.zeros(L, dtype=torch.bool))
init_prev = {"pos": torch.zeros(L, A, 3), "local": torch.zeros(L, C.local_size)}
prev = init_prev
dp, dl, dt = [], [], []
step_idx = np.linspace(0, S - 1, NSTEP).astype(int)
for step in range(S):
    raw_t = 1 - step / S
    t = float(T.sigma_scale_cosine(torch.tensor(raw_t)))
    data["t_pos"] = torch.full((L,), t); data["t_seq"] = torch.ones(L)
    if raw_t < T.PREV_THRESHOLD:
        prev = init_prev
    data["pos_noised"] = T.noise_step(data["pos"], data["t_pos"],
                                      data["batch_index"], noise[step], C.sigma_data)
    data["seq_noised"] = data["seq"]
    opt.BATCH_COND = False
    r0, p0 = m(data, prev, predict_aa=True, uniform=uni[step], compute_distogram=False)
    if step in step_idx:
        opt.BATCH_COND = True
        r1, _ = m(data, prev, predict_aa=True, uniform=uni[step], compute_distogram=False)
        opt.BATCH_COND = False
        dp.append((r0["pos"].double() - r1["pos"].double()).abs().max().item())
        dl.append((r0["local"].double() - r1["local"].double()).abs().max().item())
        dt.append((r0["trajectory"].double() - r1["trajectory"].double()).abs().max().item())
        print(f"  step {step:>3}  raw_t {raw_t:5.3f}   pos {dp[-1]:.3e}   "
              f"local {dl[-1]:.3e}   traj {dt[-1]:.3e}")
    data["pos"], prev = r0["pos"], p0
    data["seq"] = r0["aa"].argmax(dim=-1)
for nm, v in (("pos", dp), ("local", dl), ("trajectory", dt)):
    v = np.array(v)
    print(f"P7 CPU single-forward perturbation, {nm:<11} n={len(v)}  zero={int((v == 0).sum())}  "
          f"mean {v.mean():.3e}  median {np.median(v):.3e}  "
          f"p95 {np.percentile(v, 95):.3e}  max {v.max():.3e}")
