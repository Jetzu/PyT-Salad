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

__all__ = ["load_param_npz", "load_param_jax", "diffusion_block_map",
           "build_state_dict", "load_into"]

# Haiku parameter leaf names, by kind
_KINDS = {
    "lin": ("w", "b"),
    "ln": ("scale", "offset"),
    "raw": (None,),
}


# --------------------------------------------------------------------------
# reading checkpoints
# --------------------------------------------------------------------------
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
