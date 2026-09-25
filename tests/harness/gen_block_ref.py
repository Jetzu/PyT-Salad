"""Reference dump of one JAX DiffusionBlock (layer 0 weights of default_vp-200k)."""
import pickle, sys
import numpy as np
import jax, jax.numpy as jnp, haiku as hk

# XLA:GPU lowers float32 `dot` to TF32 by default, which makes a GPU-generated
# reference a ~1e-3-relative one - see notes/04_PARITY_AND_PERF.md section 5.
# Bit-exact no-op on CPU.
jax.config.update("jax_default_matmul_precision", "highest")

import salad.modules.noise_schedule_benchmark as NSB
import salad.modules.utils.geometry as G
import salad.modules.geometric.geometric as GEO
from salad.aflib.model.geometry import Vec3Array
from salad.modules.config import noise_schedule_benchmark as configs

N = 40
A = 20
rng = np.random.default_rng(7)

U = rng.uniform(1e-6, 1-1e-6, size=(N, N)).astype(np.float32)
UJ = jnp.asarray(U)

REC = {}

def patched_random(count):
    def inner(pos, item, mask, neighbours=None):
        distance = None
        if isinstance(pos, jnp.ndarray):
            assert pos.ndim == 2
            if pos.shape[0] == pos.shape[1]:
                distance = pos
            else:
                pos = Vec3Array.from_array(pos)
        if distance is None:
            distance = (pos[:, None] - pos[None, :]).norm()
        same_item = item[:, None] == item[None, :]
        weight = -3 * jnp.log(distance + 1e-6)
        gumbel = jnp.log(-jnp.log(UJ))
        weight = weight - gumbel
        distance = jnp.where(same_item, -weight, jnp.inf)
        pair_mask = mask[:, None] * mask[None, :] * same_item
        return G.get_neighbours(count)(distance, pair_mask, neighbours)
    return inner

G.get_random_neighbours = patched_random
NSB.get_random_neighbours = patched_random

_orig_en = NSB.extract_neighbours
_orig_ecn = NSB.extract_condition_neighbours
def rec_en(**kw):
    f = _orig_en(**kw)
    def inner(*a, **k):
        r = f(*a, **k); REC["current_neighbours"] = np.asarray(r); return r
    return inner
def rec_ecn(**kw):
    f = _orig_ecn(**kw)
    def inner(*a, **k):
        r = f(*a, **k); REC["cond_neighbours"] = np.asarray(r); return r
    return inner
NSB.extract_neighbours = rec_en
NSB.extract_condition_neighbours = rec_ecn

cfg = configs.default_vp

# ---- inputs -----------------------------------------------------------
resi = np.concatenate([np.arange(22), np.arange(18)]).astype(np.int32)
chain = np.concatenate([np.zeros(22), np.ones(18)]).astype(np.int32)
batch = np.zeros(N, np.int32)
mask = np.ones(N, bool); mask[-3:] = False           # exercise masking
tt = np.arange(N)[:, None]
bb = np.concatenate([3.1*np.cos(tt*1.3), 3.1*np.sin(tt*1.3), 1.4*tt], -1)
pos = (bb[:, None, :] + rng.normal(0, 1.1, (N, A, 3))).astype(np.float32)
prev_pos = (bb[:, None, :] + rng.normal(0, 1.4, (N, A, 3))).astype(np.float32)
features = rng.normal(0, 1.0, (N, 128)).astype(np.float32)
condition = rng.normal(0, 1.0, (N, 128)).astype(np.float32)
time = np.full(N, 0.4, np.float32)
dmap = np.abs(rng.normal(8.0, 3.0, (N, N))).astype(np.float32)
dmap = ((dmap + dmap.T)/2).astype(np.float32)
dmap_mask = np.zeros((N, N), bool)
dmap_mask[:6, :6] = True
np.fill_diagonal(dmap_mask, False)
chain_contacts = (chain[:, None] != chain[None, :])
flags = np.concatenate([chain_contacts[..., None], np.zeros((N, N, 2))], -1).astype(np.float32)

pair_condition = dict(dmap=jnp.asarray(dmap), dmap_mask=jnp.asarray(dmap_mask),
                      flags=jnp.asarray(flags),
                      omap=jnp.zeros((N, N, 9), jnp.float32))

# ---- load & slice params ---------------------------------------------
with open(sys.argv[1], "rb") as f:
    params = pickle.load(f)
PREFIX = "structure_diffusion/diffusion/diffusion_stack/__layer_stack_with_state/"
LAYER = 0
block_params = {}
for k, v in params.items():
    if k.startswith(PREFIX):
        newk = k[len(PREFIX):]
        block_params[newk] = {kk: np.asarray(vv)[LAYER] for kk, vv in v.items()}

def fwd(features, pos, prev_pos, condition, time):
    return NSB.DiffusionBlock(cfg)(
        features, pos, prev_pos, condition, pair_condition, time,
        jnp.asarray(resi), jnp.asarray(chain), jnp.asarray(batch), jnp.asarray(mask))

tr = hk.transform(fwd)
out_features, out_pos = tr.apply(block_params, jax.random.PRNGKey(3),
                                 jnp.asarray(features), jnp.asarray(pos),
                                 jnp.asarray(prev_pos), jnp.asarray(condition),
                                 jnp.asarray(time))

# also dump the SIPA input pair/pair_mask for the first branch
def pf(pos_):
    p, pm = NSB.diffusion_pair_features(cfg)(
        pos_, pair_condition, jnp.asarray(REC["current_neighbours"]),
        jnp.asarray(resi), jnp.asarray(chain), jnp.asarray(batch), jnp.asarray(mask))
    return p, pm
pf_t = hk.transform(pf)
# reuse the flat linear_0..6 / layer_norm / mlp params from the block
pf_params = {}
name_map = {"linear": "linear", "linear_1": "linear_1", "linear_2": "linear_2",
            "linear_3": "linear_3", "linear_4": "linear_4", "linear_5": "linear_5",
            "linear_6": "linear_6", "layer_norm": "layer_norm",
            "mlp/linear": "mlp/linear", "mlp/linear_1": "mlp/linear_1"}
for tgt, src in name_map.items():
    pf_params[tgt] = block_params["diffusion_block/" + src]
pair1, pair_mask1 = pf_t.apply(pf_params, jax.random.PRNGKey(0), jnp.asarray(pos))

out = dict(
    in_resi=resi, in_chain=chain, in_batch=batch, in_mask=mask,
    in_pos=pos, in_prev_pos=prev_pos, in_features=features,
    in_condition=condition, in_time=time,
    in_dmap=dmap, in_dmap_mask=dmap_mask, in_flags=flags,
    in_uniform=U,
    current_neighbours=REC["current_neighbours"],
    cond_neighbours=REC["cond_neighbours"],
    pair1=np.asarray(pair1), pair_mask1=np.asarray(pair_mask1),
    out_features=np.asarray(out_features), out_pos=np.asarray(out_pos),
)
np.savez(sys.argv[2], **out)
print("wrote", sys.argv[2])
for k, v in out.items():
    print(f"  {k:22s} {v.shape} {v.dtype}")
