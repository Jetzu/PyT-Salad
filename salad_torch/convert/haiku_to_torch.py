"""Load released salad (Haiku/JAX) checkpoints into the PyTorch port.

Why this is a table and not a name rewrite
------------------------------------------
Haiku names submodules by *creation order within a scope*, and plain helper
functions do not open a scope of their own. So `diffusion_pair_features` and
`update_positions` - both plain functions - dump their `Linear`s flat into the
enclosing ``diffusion_block`` scope, sharing one counter: ``linear``,
``linear_1``, ... The port groups those same layers into readable
`DiffusionPairFeatures` / `UpdatePositions` submodules, so the mapping is a
bijection but not an identity.

Worse, the flat indices are **config-dependent**. Each pair block emits:

    relpos, [dmap_mask if multi_motif], dmap, [omap if use_omap],
    flags, dist, dir, rot, pairvec, [init_dist, init_rot if init_pos]

so the per-pair-block count is ``7 + multi_motif + use_omap + 2*init_pos`` and
the position-update linear lands after two of them:

    default_vp         7 per block -> linear_14 is update_positions
    default_ve_scaled  9 per block -> linear_18
    multimotif_vp     10 per block -> linear_20

Hence `diffusion_block_map`, which derives the indices rather than hardcoding.

Weight layout: Haiku stores `Linear.w` as ``(in, out)`` and the port keeps that
layout deliberately, so **nothing is transposed here**. LayerNorm carries
``scale``/``offset``; attention carries a bare ``gamma``.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

__all__ = ["load_params", "load_param_npz", "load_param_jax", "diffusion_block_map",
           "build_state_dict", "load_into", "encoder_block_map", "encoder_map",
           "load_encoder"]

# Haiku parameter leaf names, by kind
_KINDS = {
    "lin": ("w", "b"),
    "ln": ("scale", "offset"),
    "raw": (None,),
}


# --------------------------------------------------------------------------
# reading checkpoints
# --------------------------------------------------------------------------
def load_params(path: str | Path) -> dict[str, np.ndarray]:
    """Load a checkpoint, dispatching on its extension.

    The released checkpoints are ``.jax`` pickles; ``.npz`` is the converted
    form. Callers should use this rather than picking a loader themselves -
    passing a ``.jax`` file to `load_param_npz` fails with numpy's
    "Cannot load file containing pickled data" rather than anything helpful.
    """
    path = Path(path)
    if path.suffix == ".npz":
        return load_param_npz(path)
    return load_param_jax(path)


def load_param_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load a converted ``.npz`` (flat ``module/path/leaf -> array``)."""
    data = np.load(str(path))
    return {k: data[k] for k in data.files}


def load_param_jax(path: str | Path) -> dict[str, np.ndarray]:
    """Load a raw ``.jax`` checkpoint without importing jax.

    The files are plain pickles whose arrays go through
    ``jax._src.array._reconstruct_array``, which just wraps a numpy array. A
    stubbed `find_class` is therefore enough to read them with the stdlib.
    """

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("jax"):
                if name == "_reconstruct_array":
                    def reconstruct(fn, args, state, _aval=None):
                        arr = fn(*args)
                        if state is not None:
                            arr.__setstate__(state)
                        return arr
                    return reconstruct
                return lambda *a, **k: None
            return super().find_class(module, name)

    with open(path, "rb") as fh:
        tree = _Unpickler(fh).load()

    flat: dict[str, np.ndarray] = {}

    def walk(node, prefix=""):
        for key, value in node.items():
            path_ = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(value, dict):
                walk(value, path_)
            else:
                flat[path_] = np.asarray(value)

    walk(tree)
    return flat


# --------------------------------------------------------------------------
# name maps
# --------------------------------------------------------------------------
def _pair_feature_names(multi_motif: bool, use_omap: bool, use_init_pos: bool):
    """Port attribute suffixes for one pair block, in Haiku creation order."""
    names = ["linear_relpos"]
    if multi_motif:
        names.append("linear_dmap_mask")
    names.append("linear_dmap")
    if use_omap:
        names.append("linear_omap")
    names += ["linear_flags", "linear_dist", "linear_dir", "linear_rot",
              "linear_pairvec"]
    if use_init_pos:
        names += ["linear_init_dist", "linear_init_rot"]
    return names


def _attention_map(port_prefix: str, hk_prefix: str) -> dict:
    out = {}
    pb, hb = f"{port_prefix}.ada_point_attention.", f"{hk_prefix}/ada_point_attention/"
    for name in ("layer_norm", "layer_norm_1", "layer_norm_2"):
        out[pb + name] = ("ln", hb + name)
    for name in ("qkv", "bias", "project_out"):
        out[pb + name] = ("lin", hb + name)
    out[pb + "qkv_global.linear"] = ("lin", hb + "qkv_global/linear")
    out[pb + "gamma"] = ("raw", hb + "gamma")
    return out


def diffusion_block_map(multi_motif: bool = False, use_omap: bool = False,
                        use_init_pos: bool = False) -> dict[str, tuple[str, str]]:
    """``{port_attr_path: (kind, haiku_name)}`` for one `DiffusionBlock`."""
    names = _pair_feature_names(multi_motif, use_omap, use_init_pos)
    per = len(names)
    out: dict[str, tuple[str, str]] = {}

    for block_idx, port_pair in enumerate(("pair_1", "pair_2")):
        offset = block_idx * per
        for i, attr in enumerate(names):
            idx = offset + i
            hk = "linear" if idx == 0 else f"linear_{idx}"
            out[f"{port_pair}.{attr}"] = ("lin", hk)
        ln = "layer_norm" if block_idx == 0 else "layer_norm_2"
        out[f"{port_pair}.layer_norm"] = ("ln", ln)
        mlp = "mlp" if block_idx == 0 else "mlp_1"
        out[f"{port_pair}.mlp.layers.0"] = ("lin", f"{mlp}/linear")
        out[f"{port_pair}.mlp.layers.1"] = ("lin", f"{mlp}/linear_1")

    for ln in ("layer_norm_1", "layer_norm_3", "layer_norm_4", "layer_norm_5"):
        out[ln] = ("ln", ln)

    out.update(_attention_map("attn_1", "sparse_structure_attn"))
    out.update(_attention_map("attn_2", "sparse_structure_attn_1"))

    for port, hk in (("linear_condition", "linear"),
                     ("mlp.layers.0", "mlp/linear"),
                     ("mlp.layers.1", "mlp/linear_1"),
                     ("linear_update", "linear_1"),
                     ("linear_gate", "linear_2"),
                     ("linear_chain_gate", "linear_3"),
                     ("linear_batch_gate", "linear_4"),
                     ("linear_out", "linear_5")):
        out[f"update.{port}"] = ("lin", f"light_global_update/{hk}")

    # the position-update linear is created last, after both pair blocks
    out["update_positions.linear"] = ("lin", f"linear_{2 * per}")
    return out


# --------------------------------------------------------------------------
# state dict assembly
# --------------------------------------------------------------------------
def build_state_dict(params: dict[str, np.ndarray], mapping: dict, hk_scope: str,
                     port_prefix: str = "", layer: int | None = None
                     ) -> dict[str, torch.Tensor]:
    """Turn one Haiku scope into torch state-dict entries.

    `layer` selects a slice of a `hk.LayerStack` parameter (which carries a
    leading depth axis); pass None for unstacked scopes.
    """
    scope = hk_scope.rstrip("/")
    out: dict[str, torch.Tensor] = {}
    for port, (kind, hk) in mapping.items():
        key = f"{port_prefix}{port}" if port_prefix else port
        for leaf in _KINDS[kind]:
            hk_key = f"{scope}/{hk}/{leaf}" if leaf else f"{scope}/{hk}"
            port_key = f"{key}.{leaf}" if leaf else key
            if hk_key not in params:
                raise KeyError(f"missing in checkpoint: {hk_key}")
            arr = params[hk_key]
            if layer is not None:
                arr = arr[layer]
            out[port_key] = torch.from_numpy(np.ascontiguousarray(arr))
    return out


def load_into(module: nn.Module, state: dict[str, torch.Tensor],
              strict: bool = True) -> None:
    """Load and report mismatches usefully rather than as a wall of names."""
    missing, unexpected = module.load_state_dict(state, strict=strict)
    if missing or unexpected:
        raise RuntimeError(
            f"state dict mismatch: {len(missing)} missing "
            f"(e.g. {list(missing)[:3]}), {len(unexpected)} unexpected "
            f"(e.g. {list(unexpected)[:3]})")


def load_diffusion_stack(stack: nn.Module, params: dict[str, np.ndarray],
                         config, strict: bool = True) -> None:
    """Load all `diffusion_depth` layers of a `DiffusionStack`."""
    scope = ("structure_diffusion/diffusion/diffusion_stack/"
             "__layer_stack_with_state/diffusion_block")
    mapping = diffusion_block_map(
        multi_motif=bool(config.multi_motif),
        use_omap=bool(config.use_omap),
        use_init_pos=(config.preconditioning == "edm_scaled") or bool(config.multi_motif),
    )
    state: dict[str, torch.Tensor] = {}
    for layer in range(config.diffusion_depth):
        state.update(build_state_dict(params, mapping, scope,
                                      port_prefix=f"blocks.{layer}.", layer=layer))
    load_into(stack, state, strict=strict)


# --------------------------------------------------------------------------
# the remaining scopes
# --------------------------------------------------------------------------
_SD = "structure_diffusion"


def condition_map() -> dict[str, tuple[str, str]]:
    """`Condition`, at ``~prepare_condition/condition``."""
    return {
        "linear_aa": ("lin", "linear"),
        "layer_norm_aa": ("ln", "layer_norm"),
        "linear_dssp": ("lin", "linear_1"),
        "layer_norm_dssp": ("ln", "layer_norm_1"),
        "linear_dssp_mean": ("lin", "linear_2"),
        "layer_norm_dssp_mean": ("ln", "layer_norm_2"),
    }


def aa_decoder_block_map() -> dict[str, tuple[str, str]]:
    """One `AADecoderBlock`. Its pair features are a plain function upstream,
    so they share the block's flat counter: linear..linear_5."""
    out = {
        "pair.linear_relpos": ("lin", "linear"),
        "pair.linear_dist": ("lin", "linear_1"),
        "pair.linear_dir": ("lin", "linear_2"),
        "pair.linear_rot": ("lin", "linear_3"),
        "pair.linear_pairvec": ("lin", "linear_4"),
        # the LayerNorm is constructed before its argument, so it takes index 0
        "pair.layer_norm_aa": ("ln", "layer_norm"),
        "pair.linear_aa": ("lin", "linear_5"),
        "pair.layer_norm": ("ln", "layer_norm_1"),
        "pair.mlp.layers.0": ("lin", "mlp/linear"),
        "pair.mlp.layers.1": ("lin", "mlp/linear_1"),
        "layer_norm_2": ("ln", "layer_norm_2"),
        "layer_norm_3": ("ln", "layer_norm_3"),
    }
    out.update(_attention_map("attn", "sparse_structure_attn"))
    for port, hk in (("mlp.layers.0", "mlp/linear"), ("mlp.layers.1", "mlp/linear_1"),
                     ("linear_update", "linear"), ("linear_gate", "linear_1"),
                     ("linear_out", "linear_2")):
        out[f"update.{port}"] = ("lin", f"light_global_update/{hk}")
    return out


def distogram_decoder_map() -> dict[str, tuple[str, str]]:
    return {
        "linear_relpos": ("lin", "linear"),
        "linear_dist": ("lin", "linear_1"),
        "linear_dir": ("lin", "linear_2"),
        "linear_rot": ("lin", "linear_3"),
        "linear_pairvec": ("lin", "linear_4"),
        "linear_local_self": ("lin", "linear_5"),
        "linear_local_nb": ("lin", "linear_6"),
        "layer_norm": ("ln", "layer_norm"),
        "mlp.layers.0": ("lin", "mlp/linear"),
        "mlp.layers.1": ("lin", "mlp/linear_1"),
    }


def encoder_block_map() -> dict[str, tuple[str, str]]:
    """One `EncoderBlock` (`salad_torch/modules/encoder.py`).

    `encoder_pair_features` is a plain function upstream, so - exactly like
    `aa_decoder_pair_features` (`aa_decoder_block_map` above) - it shares the
    enclosing `encoder_block` scope's flat `linear`/`layer_norm` counters:
    `linear`..`linear_4`, `mlp`, then its own final LayerNorm takes the first
    bare `layer_norm` name in the scope. The block's own two LayerNorms come
    next in creation order (`layer_norm_1` before attention, `layer_norm_2`
    before the update) - one lower than `aa_decoder_block_map`'s
    `layer_norm_2`/`layer_norm_3`, because `AADecoderPairFeatures` itself
    consumes two bare LayerNorm names (`layer_norm`, `layer_norm_1`) that
    `EncoderPairFeatures` does not (no aa embedding term).
    """
    out = {
        "pair_features.linear_relpos": ("lin", "linear"),
        "pair_features.linear_dist": ("lin", "linear_1"),
        "pair_features.linear_dir": ("lin", "linear_2"),
        "pair_features.linear_rot": ("lin", "linear_3"),
        "pair_features.linear_pairvec": ("lin", "linear_4"),
        "pair_features.layer_norm": ("ln", "layer_norm"),
        "pair_features.mlp.layers.0": ("lin", "mlp/linear"),
        "pair_features.mlp.layers.1": ("lin", "mlp/linear_1"),
        "layer_norm_1": ("ln", "layer_norm_1"),
        "layer_norm_2": ("ln", "layer_norm_2"),
    }
    out.update(_attention_map("attn", "sparse_structure_attn"))
    for port, hk in (("mlp.layers.0", "mlp/linear"), ("mlp.layers.1", "mlp/linear_1"),
                     ("linear_update", "linear"), ("linear_gate", "linear_1"),
                     ("linear_out", "linear_2")):
        out[f"update.{port}"] = ("lin", f"light_global_update/{hk}")
    return out


def encoder_map() -> dict[str, tuple[str, str]]:
    """Everything in the `encoder` scope directly, plus `~prepare_features`
    (both outside the `self_cond_stack` layer-stack - see `load_encoder`)."""
    return {
        "vector_linear": ("lin", "~prepare_features/vector_linear"),
        "local_mlp.layers.0": ("lin", "~prepare_features/mlp/linear"),
        "local_mlp.layers.1": ("lin", "~prepare_features/mlp/linear_1"),
        "local_layer_norm": ("ln", "~prepare_features/layer_norm"),
        # position-update MLP, created directly in `encoder` (§4/§6.4) -
        # bias=True and NOT zero-init, see Encoder's docstring
        "update_mlp.layers.0": ("lin", "mlp/linear"),
        "update_mlp.layers.1": ("lin", "mlp/linear_1"),
    }


def load_encoder(encoder: nn.Module, params: dict[str, np.ndarray], config,
                 strict: bool = True) -> dict[str, torch.Tensor]:
    """Load `structure_diffusion/~encode/encoder/*` into a port `Encoder`.

    Present in every released checkpoint (59 leaves, identical across all 4
    configs - none of them touch `augment_size`/`encoder_depth`/`resi_dual`/
    `equivariance`), but `load_model` above never loads it: the Encoder builds
    the training target `data["pos"]`, and de-novo sampling never calls it (no
    atom14 ground truth to encode at sampling time). Separate entry point so a
    caller that only needs to sample is not forced to also carry these weights.
    """
    scope = f"{_SD}/~encode/encoder"
    state: dict[str, torch.Tensor] = {}

    state.update(build_state_dict(params, encoder_map(), scope))

    block_scope = f"{scope}/self_cond_stack/__layer_stack_no_state/encoder_block"
    for layer in range(config.encoder_depth):
        state.update(build_state_dict(
            params, encoder_block_map(), block_scope,
            port_prefix=f"encoder_stack.blocks.{layer}.", layer=layer))

    # self_cond_stack's own final LayerNorm, outside the layer-stack
    state.update(build_state_dict(
        params, {"layer_norm": ("ln", "layer_norm")}, f"{scope}/self_cond_stack",
        port_prefix="encoder_stack."))

    load_into(encoder, state, strict=strict)
    return state


def load_model(model: nn.Module, params: dict[str, np.ndarray], config,
               strict: bool = True) -> dict[str, torch.Tensor]:
    """Load a full `StructureDiffusionPredict` from a flat Haiku param dict.

    Returns the assembled state dict (useful for inspection) after loading it.
    """
    state: dict[str, torch.Tensor] = {}

    state.update(build_state_dict(
        params, condition_map(), f"{_SD}/~prepare_condition/condition",
        port_prefix="prepare_condition.condition."))

    d = "diffusion."
    dif = f"{_SD}/diffusion"

    # trunk
    trunk_map = diffusion_block_map(
        multi_motif=bool(config.multi_motif), use_omap=bool(config.use_omap),
        use_init_pos=(config.preconditioning == "edm_scaled") or bool(config.multi_motif))
    trunk_scope = f"{dif}/diffusion_stack/__layer_stack_with_state/diffusion_block"
    for layer in range(config.diffusion_depth):
        state.update(build_state_dict(
            params, trunk_map, trunk_scope,
            port_prefix=f"{d}diffusion_stack.blocks.{layer}.", layer=layer))

    # prepare_features
    pf = f"{dif}/~prepare_features"
    state.update(build_state_dict(params, {
        "prepare_mlp.layers.0": ("lin", "mlp/linear"),
        "prepare_mlp.layers.1": ("lin", "mlp/linear_1"),
        "prepare_layer_norm_prev": ("ln", "layer_norm"),
        "prepare_layer_norm_local": ("ln", "layer_norm_1"),
        "prepare_layer_norm_cond": ("ln", "layer_norm_2"),
    }, pf, port_prefix=d))

    # aa decoder
    aa = f"{dif}/aa_decoder"
    state.update(build_state_dict(params, {
        "linear_aa_embed": ("lin", "linear"),
        "layer_norm": ("ln", "layer_norm"),
        "linear_out": ("lin", "linear_1"),
    }, aa, port_prefix=f"{d}aa_decoder."))
    state.update(build_state_dict(params, {"layer_norm": ("ln", "layer_norm")},
                                  f"{aa}/aa_decoder_stack",
                                  port_prefix=f"{d}aa_decoder.aa_decoder_stack."))
    aa_block_scope = f"{aa}/aa_decoder_stack/__layer_stack_no_state/aa_decoder_block"
    for layer in range(config.aa_decoder_depth):
        state.update(build_state_dict(
            params, aa_decoder_block_map(), aa_block_scope,
            port_prefix=f"{d}aa_decoder.aa_decoder_stack.blocks.{layer}.", layer=layer))

    # heads
    state.update(build_state_dict(params, distogram_decoder_map(),
                                  f"{dif}/distogram_decoder",
                                  port_prefix=f"{d}distogram_decoder."))
    state.update(build_state_dict(params, {
        "dssp_decoder": ("lin", "linear"),
        "angle_head.mlp.layers.0": ("lin", "mlp/linear"),
        "angle_head.mlp.layers.1": ("lin", "mlp/linear_1"),
    }, dif, port_prefix=d))

    load_into(model, state, strict=strict)
    return state
