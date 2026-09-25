"""R05 (JAX side): run the UPSTREAM Encoder with real default_vp-200k weights
and dump inputs/outputs. Run inside salad.sif."""
import pickle, sys
import numpy as np, jax, jax.numpy as jnp, haiku as hk
import salad.modules.noise_schedule_benchmark as NSB
from salad.modules.config import noise_schedule_benchmark as configs

cfg = configs.default_vp
cfg.eval = True

class SD(hk.Module):
    def __init__(self, c):
        super().__init__(name="structure_diffusion")
        self.config = c
    def encode(self, data):
        return NSB.Encoder(self.config)(data)

def fwd(data):
    return SD(cfg).encode(data)

f = hk.transform(fwd)
with open("/hnvme/workspace/v181bb10-salad/params/default_vp-200k.jax", "rb") as fh:
    params = pickle.load(fh)

rng = np.random.default_rng(3)
def build(n, chains=1):
    t = np.arange(n)[:, None]
    bb = np.concatenate([2.3*np.cos(t*1.7), 2.3*np.sin(t*1.7), 1.5*t], -1)
    pos = (bb[:, None, :] + rng.normal(0, 0.8, (n, 5, 3))).astype(np.float32)
    ch = (np.arange(n) // max(1, n // chains)).clip(max=chains-1).astype(np.int32)
    return dict(pos=pos, residue_index=np.arange(n, dtype=np.int32),
                chain_index=ch, batch_index=np.zeros(n, np.int32),
                mask=np.ones(n, bool))

out = {}
for tag, n, ch in (("L64c1", 64, 1), ("L96c3", 96, 3), ("L16c1", 16, 1)):
    d = build(n, ch)
    local, pos = f.apply(params, jax.random.PRNGKey(0), {k: jnp.asarray(v) for k, v in d.items()})
    for k, v in d.items():
        out[f"{tag}|in_{k}"] = np.asarray(v)
    out[f"{tag}|out_local"] = np.asarray(local)
    out[f"{tag}|out_pos"] = np.asarray(pos)
    print(tag, "local", np.asarray(local).shape, "pos", np.asarray(pos).shape,
          "|pos|max", float(np.abs(np.asarray(pos)).max()))
np.savez("/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N003-red-team-pass-fast-path-symmetry-encoder/r05_encoder_ref.npz", **out)
print("wrote r05_encoder_ref.npz")
