import sys
import numpy as np, jax, jax.numpy as jnp, haiku as hk
import salad.modules.noise_schedule_benchmark as NSB
from salad.modules.basic import Linear, MLP
from salad.aflib.model.geometry import Vec3Array
from salad.modules.utils.geometry import sequence_relative_position, distance_rbf
from salad.modules.geometric import (distance_features, direction_features,
                                     position_rotation_features, pair_vector_features)
from salad.modules.config import noise_schedule_benchmark as configs

D = np.load(sys.argv[1])
NPZ = np.load(sys.argv[2])
P = "structure_diffusion/diffusion/diffusion_stack/__layer_stack_with_state/diffusion_block/"
L = 0
def par(scope):
    out = {}
    for nm in ("w","b","scale","offset"):
        k = P + scope + "/" + nm
        if k in NPZ: out[nm] = jnp.asarray(NPZ[k][L])
    return out

cfg = configs.default_vp
pos = jnp.asarray(D["in_pos"]); nb = jnp.asarray(D["current_neighbours"])
resi = jnp.asarray(D["in_resi"]); chain = jnp.asarray(D["in_chain"])
batch = jnp.asarray(D["in_batch"]); mask = jnp.asarray(D["in_mask"])
idx = jnp.arange(pos.shape[0], dtype=jnp.int32)
pc = dict(dmap=jnp.asarray(D["in_dmap"]), dmap_mask=jnp.asarray(D["in_dmap_mask"]),
          flags=jnp.asarray(D["in_flags"]))
pcn = jax.tree_util.tree_map(lambda x: x[idx[:, None], nb], pc)

out = {}
out["f_relpos"] = np.asarray(sequence_relative_position(32, one_hot=True, cyclic=False)(resi, chain, batch, nb))
out["f_dmap"]   = np.asarray(jnp.where(pcn["dmap_mask"][..., None], distance_rbf(pcn["dmap"]), 0))
out["f_flags"]  = np.asarray(jnp.where(pcn["flags"].any(axis=-1)[..., None], pcn["flags"], 0))
V = Vec3Array.from_array(pos)
out["f_dist"]   = np.asarray(distance_features(V, nb, d_min=0.0, d_max=22.0))
out["f_dir"]    = np.asarray(direction_features(V, nb))
out["f_rot"]    = np.asarray(position_rotation_features(V, nb))
out["f_pairvec"]= np.asarray(pair_vector_features(V, nb))

names = ["linear","linear_1","linear_2","linear_3","linear_4","linear_5","linear_6"]
feats = [out["f_relpos"], out["f_dmap"], out["f_flags"], out["f_dist"],
         out["f_dir"], out["f_rot"], out["f_pairvec"]]
acc = jnp.zeros((pos.shape[0], nb.shape[1], 64), jnp.float32)
for i, (nm, f) in enumerate(zip(names, feats)):
    w = par(nm)["w"]
    acc = acc + jnp.einsum("...k,ki->...i", jnp.asarray(f), w)
    out[f"acc_{i}"] = np.asarray(acc)

def ln_fn(x):
    return hk.LayerNorm([-1], True, True)(x)
lnt = hk.transform(ln_fn)
ln_p = {"layer_norm": par("layer_norm")}
after_ln = lnt.apply(ln_p, None, acc)
out["after_ln"] = np.asarray(after_ln)

def mlp_fn(x):
    return MLP(128, 64, activation=jax.nn.gelu, final_init="linear")(x)
mt = hk.transform(mlp_fn)
mp = {"mlp/linear": par("mlp/linear"), "mlp/linear_1": par("mlp/linear_1")}
out["after_mlp"] = np.asarray(mt.apply(mp, None, after_ln))
np.savez(sys.argv[3], **out)
print("wrote", sys.argv[3])
