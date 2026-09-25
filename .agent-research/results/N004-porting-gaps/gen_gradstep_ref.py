"""Task 3: dump JAX gradients + one optax Adam step for the REAL training
step (StructureDiffusion's forward+loss composition, predict_aa=False - NOT
StructureDiffusionPredict's predict_aa=True inference variant), so the torch
port's `training_step` can be checked against it.

Almost identical scaffolding to tests/harness/gen_loss_ref2.py (same
build_data, same install_fixed_uniform trick) - reused deliberately, since
that fixture already matches JAX bit-for-bit on the forward+loss values
(check_loss2.py, 2e-7). The one behavioural difference from that file: this
composes `StructureDiffusionInference.prepare_condition` (deterministic,
"nothing revealed" - no aa_condition/dssp_condition supplied) with
`Diffusion.__call__(..., predict_aa=False)` directly, matching what the real
upstream training entry point `StructureDiffusion.__call__` actually calls
(NSB:412-439: `result = diffusion(data, prev)`, predict_aa defaults False),
rather than `StructureDiffusionPredict` (always predict_aa=True, an
inference-only wrapper). Self-conditioning is pinned OFF (prev=zero) rather
than replaying NSB's `count = randint(0,2)` gate, so the comparison is fully
deterministic without needing to reproduce JAX's threefry stream on the torch
side (torch and JAX RNGs are not interoperable regardless of seed).

Run inside the container:
    apptainer run --bind /hnvme:/hnvme --bind $PWD:$PWD \
        --bind /home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research:/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research \
        salad.sif .agent-research/results/N004-porting-gaps/gen_gradstep_ref.py \
        params/default_vp-200k.jax .agent-research/results/N004-porting-gaps/gradstep_default_vp.npz
"""
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

import salad.modules.noise_schedule_benchmark as NSB
import salad.modules.utils.geometry as G
from salad.aflib.common import residue_constants as RC
from salad.aflib.model.geometry import Vec3Array
from salad.modules.config import noise_schedule_benchmark as configs

_FIELDS = {}

# fixed hyperparameters for the one optimizer step (spec-04 §B.2/§B.3 values,
# minus the schedule - a CONSTANT lr is used here since only one step runs)
LR = 1e-3
B1, B2 = 0.9, 0.99
ADAM_EPS = 1e-9
CLIP = 0.1


def install_fixed_uniform(name):
    u = jnp.asarray(_FIELDS[name])

    def patched(count):
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
            gumbel = jnp.log(-jnp.log(u))
            weight = weight - gumbel
            distance = jnp.where(same_item, -weight, jnp.inf)
            pair_mask = mask[:, None] * mask[None, :] * same_item
            return G.get_neighbours(count)(distance, pair_mask, neighbours)
        return inner

    G.get_random_neighbours = patched
    NSB.get_random_neighbours = patched


def build_data(rng, augment_size=15):
    """Verbatim from tests/harness/gen_loss_ref2.py (already-proven fixture:
    2 batch items, 3 chains, padding, mixed t_pos)."""
    lens = [(0, [40]), (1, [30, 26])]
    resi, chain, batch = [], [], []
    for b, chains in lens:
        for ci, cl in enumerate(chains):
            resi.append(np.arange(cl, dtype=np.int32))
            chain.append(np.full(cl, ci, np.int32))
            batch.append(np.full(cl, b, np.int32))
    resi = np.concatenate(resi)
    chain = np.concatenate(chain)
    batch = np.concatenate(batch)
    n = resi.shape[0]
    a = 5 + augment_size

    mask = np.ones(n, bool)
    mask[-6:] = False

    pos = np.zeros((n, a, 3), np.float32)
    atom_pos = np.zeros((n, 14, 3), np.float32)
    off = 0
    for b, chains in lens:
        for ci, cl in enumerate(chains):
            t = np.arange(cl)[:, None]
            ca = np.concatenate(
                [2.3 * np.cos(t * 1.7), 2.3 * np.sin(t * 1.7), 1.5 * t], -1)
            ca = ca + np.array([[40.0 * b, 25.0 * ci, 0.0]])
            blob = ca[:, None, :] + rng.normal(0, 1.2, (cl, a, 3))
            pos[off:off + cl] = blob.astype(np.float32)
            ideal = np.array([[-1.46, 0.0, 0.0], [0.0, 0.0, 0.0],
                              [0.53, 1.43, 0.0], [0.63, 2.62, 0.25]], np.float32)
            ap = np.zeros((cl, 14, 3), np.float32)
            ap[:, :4] = ca[:, None, :] + ideal[None]
            ap[:, 4:] = ca[:, None, :] + rng.normal(0, 2.0, (cl, 10, 3))
            atom_pos[off:off + cl] = ap
            off += cl

    # ALL-MASK (20), unlike gen_loss_ref2.py's real random aa_gt: `AADecoder.
    # train_forward` (torch: decoders.py:188) draws its OWN per-batch-item
    # masking rate + per-residue Bernoulli ("ADM" corruption) whenever aa_gt
    # is not already all-mask, on BOTH backends independently, with no
    # override hook on either side (unlike the Gumbel neighbour draw, which
    # both `Diffusion.forward` and this script's `install_fixed_uniform`
    # DO let a caller pin). Two independent RNG streams (JAX threefry vs
    # torch's generator) cannot be made to draw the SAME random mask, so with
    # a real aa_gt the "aa" loss term (and, via the shared trunk `local`/`pos`
    # features feeding aa_decoder, a large fraction of EVERY other
    # parameter's gradient too - the mask choice is discrete and non-smooth)
    # would differ between backends for a reason that has nothing to do with
    # port correctness. `train_forward` short-circuits to a deterministic
    # zero-corruption no-op when `(aa == 20).all()` - the exact convention
    # `scripts/gen_fixtures_model.py`'s `build_data` already uses for this
    # same reason - so this fixture uses it too, and the aa_decoder's own
    # gradient path is consequently OUT OF SCOPE for this check (see the
    # task report).
    aa_gt = np.full(n, 20, np.int32)

    canonical = np.asarray(RC.restype_atom14_mask)[aa_gt]
    drop = rng.random((n, 14)) < 0.10
    atom_mask = (canonical * (~drop)).astype(np.float32)

    t_pos = np.where(batch == 0, 0.30, 0.45).astype(np.float32)

    return dict(
        pos=pos.astype(np.float32),
        atom_pos=atom_pos,
        atom_mask=atom_mask,
        aa_gt=aa_gt,
        aa_mask=(rng.random(n) < 0.2),
        dssp_gt=rng.integers(0, 3, n).astype(np.int32),
        dssp_mask=(rng.random(n) < 0.2),
        seq=np.full(n, 20, np.int32),
        residue_index=resi,
        chain_index=chain,
        batch_index=batch,
        mask=mask,
        cyclic_mask=np.zeros(n, bool),
        t_pos=t_pos,
        t_seq=np.full(n, 1.0, np.float32),
    )


def flatten_params(tree):
    """Haiku params/grads pytree {module: {leaf: array}} -> flat
    {f"{module}/{leaf}": np.ndarray}, matching build_state_dict's key format
    (haiku_to_torch.py) so the SAME mapping tables can consume it."""
    out = {}
    for module, leafs in tree.items():
        for leaf, arr in leafs.items():
            out[f"{module}/{leaf}"] = np.asarray(arr)
    return out


def main(param_path, out_path, config_name="default_vp"):
    rng = np.random.default_rng(20260925)
    config = getattr(configs, config_name)
    config.eval = True

    with open(param_path, "rb") as f:
        params = pickle.load(f)

    data = build_data(rng, config.augment_size)
    n = data["mask"].shape[0]
    _FIELDS["model"] = rng.uniform(1e-6, 1 - 1e-6, (n, n)).astype(np.float32)
    _FIELDS["loss"] = rng.uniform(1e-6, 1 - 1e-6, (n, n)).astype(np.float32)

    out = {f"in_{k}": v for k, v in data.items()}
    out["uniform_model"] = _FIELDS["model"]
    out["uniform_loss"] = _FIELDS["loss"]

    prev = dict(pos=np.zeros_like(data["pos"]),
               local=np.zeros((n, config.local_size), np.float32))

    # ---- 1. noising (fixes pos_noised/seq_noised) --------------------------
    noise_t = hk.transform(lambda d: NSB.StructureDiffusionNoise(config)(d))
    jdata = jax.tree_util.tree_map(jnp.asarray, data)
    noised = noise_t.apply({}, jax.random.PRNGKey(0), jdata)
    for k, v in noised.items():
        out[f"noised_{k}"] = np.asarray(v)

    full_data = dict(jdata)
    full_data.update(noised)
    jprev = jax.tree_util.tree_map(jnp.asarray, prev)

    # ---- 2. the REAL training step: prepare_condition (deterministic,
    #    nothing revealed) + Diffusion(predict_aa=False) + Diffusion.loss ----
    install_fixed_uniform("model")

    class _TrainPredict(NSB.StructureDiffusionPredict):
        """`StructureDiffusionPredict.__call__` verbatim, with the one
        hardcoded `predict_aa=True` flipped to `False` - matching the REAL
        upstream training entry point `StructureDiffusion.__call__`
        (NSB:412-439: `result = diffusion(data, prev)`, predict_aa defaults
        False), not the inference-only `StructureDiffusionPredict` this
        subclasses. Copied as a subclass (not a free function) specifically
        so `self.prepare_condition`/`diffusion = Diffusion(c)` open the SAME
        Haiku scope path (``structure_diffusion/...``) the checkpoint uses -
        constructing `Diffusion(config)` outside of a module named
        "structure_diffusion" would silently look up the wrong scope.
        """
        def __call__(self, data, prev):
            c = self.config
            diffusion = NSB.Diffusion(c)
            data.update(self.prepare_condition(data))
            if prev is None:
                prev = diffusion.init_prev(data)
            result = diffusion(data, prev, predict_aa=False)
            new_prev = dict(pos=result["pos"], local=result["local"])
            return result, new_prev

    def step_fn(d, p):
        return _TrainPredict(config)(dict(d), p)

    def loss_fn(params, key, d, p):
        result, new_prev = hk.transform(step_fn).apply(params, key, d, p)
        install_fixed_uniform("loss")
        total, losses = hk.transform(
            lambda dd, rr: NSB.Diffusion(config).loss(dd, rr)
        ).apply({}, jax.random.PRNGKey(11), d, result)
        return total, (losses, result, new_prev)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total, (losses, result, new_prev)), grads = grad_fn(
        params, jax.random.PRNGKey(7), full_data, jprev)

    out["loss_total"] = np.asarray(total)
    for k, v in losses.items():
        out[f"loss_{k}"] = np.asarray(v)
        print(f"  loss/{k:22s} {float(v): .10f}")
    print(f"  loss/{'TOTAL':22s} {float(total): .10f}")

    for k, v in result.items():
        try:
            out[f"res_{k}"] = np.asarray(v)
        except Exception:
            pass

    # ---- 3. flatten + save gradients ---------------------------------------
    grad_flat = flatten_params(grads)
    gnorm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in
                               jax.tree_util.tree_leaves(grads))))
    print(f"  global grad norm (pre-clip) = {gnorm:.6f}")
    for k, v in grad_flat.items():
        out[f"grad_{k}"] = v.astype(np.float32)

    # ---- 4. one optax Adam step, matching spec-04 sB.2 EXACTLY -------------
    optimizer = optax.chain(
        optax.clip_by_global_norm(CLIP),
        optax.scale_by_adam(B1, B2, eps=ADAM_EPS),
        optax.scale_by_schedule(lambda count: LR),
        optax.scale(-1.0),
    )
    opt_state = optimizer.init(params)
    updates, _ = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    new_flat = flatten_params(new_params)
    for k, v in new_flat.items():
        out[f"after_{k}"] = v.astype(np.float32)

    out["lr"] = np.float32(LR)
    out["b1"] = np.float32(B1)
    out["b2"] = np.float32(B2)
    out["adam_eps"] = np.float32(ADAM_EPS)
    out["clip"] = np.float32(CLIP)
    out["global_grad_norm_preclip"] = np.float32(gnorm)

    np.savez(out_path, **out)
    print(f"wrote {out_path}: {len(out)} arrays (n={n}, "
         f"{len(grad_flat)} grad leaves)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
