"""R05 (torch side): FIRST-EVER numerical parity check of the ported Encoder."""
import sys, numpy as np, torch
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch")
from salad_torch.config import get_config
from salad_torch.convert.haiku_to_torch import load_param_npz, load_encoder
from salad_torch.modules.encoder import Encoder
from salad_torch.geometry.rigid import Rigid as _R
# shim ONLY so the run reaches the numerics; the missing __getitem__ is defect D1
if not hasattr(_R, "__getitem__"):
    _R.__getitem__ = lambda self, key: self.unsqueeze_batch(1)
    print("SHIM: Rigid.__getitem__ restored (defect D1 workaround)")

torch.set_grad_enabled(False)
cfg = get_config("default_vp")
enc = Encoder(cfg).eval()
load_encoder(enc, load_param_npz("/hnvme/workspace/v181bb10-salad/default_vp-200k.npz"), cfg, strict=True)

D = np.load("/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N003-redteam/r05_encoder_ref.npz")
tags = sorted({k.split("|")[0] for k in D.files})
ok_all = True
for tag in tags:
    d = dict(pos=torch.from_numpy(D[f"{tag}|in_pos"]),
             residue_index=torch.from_numpy(D[f"{tag}|in_residue_index"]).long(),
             chain_index=torch.from_numpy(D[f"{tag}|in_chain_index"]).long(),
             batch_index=torch.from_numpy(D[f"{tag}|in_batch_index"]).long(),
             mask=torch.from_numpy(D[f"{tag}|in_mask"]))
    local, pos = enc(d)
    for name, got, ref in (("local", local.numpy(), D[f"{tag}|out_local"]),
                           ("pos",   pos.numpy(),   D[f"{tag}|out_pos"])):
        den = max(float(np.abs(ref).max()), 1e-12)
        rel = float(np.abs(got - ref).max()) / den
        ok = rel <= 2e-5; ok_all &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {tag} {name:<6} rel={rel:.3e}  max|d|={np.abs(got-ref).max():.3e}")
    dev = np.linalg.norm(pos.numpy() - D[f"{tag}|out_pos"], axis=-1)
    ok = dev.max() <= 1e-3; ok_all &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {tag} coords mean={dev.mean():.5f} A max={dev.max():.5f} A")
    # per-atom breakdown: backbone 0:5 vs augment 5:20
    print(f"       backbone[0:5] max dev {np.linalg.norm(pos.numpy()[:, :5]-D[f'{tag}|out_pos'][:, :5],axis=-1).max():.5f} A"
          f"   augment[5:20] max dev {np.linalg.norm(pos.numpy()[:, 5:]-D[f'{tag}|out_pos'][:, 5:],axis=-1).max():.5f} A")
print("\nOK" if ok_all else "\nFAILED")
