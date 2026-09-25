"""Instrumented JAX forward: dump every intermediate of DiffusionBlock 0..5.

Usage (inside the container):
    apptainer run ... jax_dump.py <config> <params.jax> <out.npz> [length]
"""
import os, pickle, sys
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts"))
import gen_fixtures_model as GEN

import salad.modules.noise_schedule_benchmark as NSB
import salad.modules.utils.geometry as G
from salad.aflib.model.geometry import Vec3Array
from salad.modules.config import noise_schedule_benchmark as configs

REC = {}


def _mk(name):
    def f(x):
        REC.setdefault(name, []).append(np.asarray(x))
    return f


def rec(name, x):
    jax.debug.callback(_mk(name), x)
    return x


class RecBlock(NSB.DiffusionBlock):
    def __call__(self, features, pos, prev_pos,
                         condition, pair_condition, time,
                         resi, chain, batch, mask,
                         cyclic_mask=None, sup_neighbours=None, init_pos=None):
      c = self.config
      residual_update, residual_input, _ = NSB.get_residual_gadgets(features, c.resi_dual)
      rec("block_in_features", features)
      rec("block_in_pos", pos)
      rec("block_condition", condition)
      rec("block_prev_pos", prev_pos)
      current_neighbours = NSB.extract_neighbours(
          num_index=16, num_spatial=16, num_random=32)(
              Vec3Array.from_array(pos), resi, chain, batch, mask)
      rec("current_neighbours", current_neighbours)
      cond_neighbours = NSB.extract_condition_neighbours(
          num_index=16, num_spatial=16, num_random=16, num_block=16)(
              Vec3Array.from_array(prev_pos), pair_condition, resi, chain, batch, mask)
      rec("cond_neighbours", cond_neighbours)

      pair_feature_function = NSB.diffusion_pair_features
      attention = NSB.SparseStructureAttention

      pair, pair_mask = pair_feature_function(c)(
          pos, pair_condition, current_neighbours,
          resi, chain, batch, mask, cyclic_mask=cyclic_mask, init_pos=init_pos)
      rec("pair_1", pair)
      ln1 = residual_input(features)
      rec("ln_1", ln1)
      att1 = attention(c)(ln1, pos / c.sigma_data, pair, pair_mask,
                          current_neighbours, resi, chain, batch, mask)
      rec("attn_1", att1)
      features = residual_update(features, att1)

      pair, pair_mask = pair_feature_function(c)(
          prev_pos, pair_condition, cond_neighbours,
          resi, chain, batch, mask, cyclic_mask=cyclic_mask, init_pos=init_pos)
      rec("pair_2", pair)
      ln3 = residual_input(features)
      rec("ln_3", ln3)
      att2 = attention(c)(ln3, prev_pos / c.sigma_data, pair, pair_mask,
                          cond_neighbours, resi, chain, batch, mask)
      rec("attn_2", att2)
      features = residual_update(features, att2)

      ln4 = residual_input(features)
      rec("ln_4", ln4)
      upd = NSB.Update(c, global_update=True)(ln4, pos, chain, batch, mask, condition)
      rec("update_out", upd)
      features = residual_update(features, upd)

      local_norm = residual_input(features)
      rec("local_norm", local_norm)
      out_scale = NSB.preconditioning_scale_factors(c, time[:, None], c.sigma_data)["out"]
      pos = NSB.update_positions(pos, local_norm, scale=out_scale, symm=c.symm)
      rec("block_out_pos", pos)
      rec("block_out_features", features)
      return features, pos.astype(local_norm.dtype)


def main(config_name, param_path, out_path, length=48, inject=None):
    rng = np.random.default_rng(1234)
    config = getattr(configs, config_name)
    config.eval = True
    with open(param_path, "rb") as f:
        params = pickle.load(f)

    data = GEN.build_data(length, config.augment_size, rng,
                          multi_motif=bool(config.multi_motif))

    prev = dict(pos=np.zeros_like(data["pos"]),
                local=np.zeros((length, config.local_size), np.float32))

    def noise_fn(d):
        return NSB.StructureDiffusionNoise(config)(d)
    noise_t = hk.transform(noise_fn)
    jdata = jax.tree_util.tree_map(jnp.asarray, data)
    noised = noise_t.apply({}, jax.random.PRNGKey(0), jdata)
    if inject is not None:
        F = np.load(inject)
        noised = {k: jnp.asarray(F[f"noised_{k}"]) for k in noised}
        print("INJECTED noised_* from", inject)

    GEN.install_fixed_uniform(length)
    NSB.DiffusionBlock = RecBlock

    fixed = dict(jdata)
    fixed.update(noised)

    def predict_fn(d, p):
        return NSB.StructureDiffusionPredict(config)(d, p)
    predict_t = hk.transform(predict_fn)
    jprev = jax.tree_util.tree_map(jnp.asarray, prev)
    result, new_prev = predict_t.apply(params, jax.random.PRNGKey(7), fixed, jprev)
    jax.block_until_ready(result)

    out = {}
    for k, vs in REC.items():
        for i, v in enumerate(vs):
            out[f"{k}#{i}"] = v
    out["final_pos"] = np.asarray(result["pos"])
    out["final_local"] = np.asarray(result["local"])
    np.savez(out_path, **out)
    print(f"wrote {out_path}")
    for k in sorted(REC):
        print(f"  {k:24s} x{len(REC[k])}  {REC[k][0].shape} {REC[k][0].dtype}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         int(sys.argv[4]) if len(sys.argv) > 4 else 48,
         sys.argv[5] if len(sys.argv) > 5 else None)
