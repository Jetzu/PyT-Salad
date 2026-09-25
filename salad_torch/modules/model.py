"""Top-level model: conditioning + noising + the denoiser.

Haiku scope ``structure_diffusion``. Upstream splits the same parameters across
several classes that all share that name, so one checkpoint serves all of them:

  StructureDiffusion          training (loss)
  StructureDiffusionInference noises internally, then denoises
  StructureDiffusionNoise     the noising step alone (no parameters)
  StructureDiffusionPredict   denoise only - what the composable API uses

Here `StructureDiffusionPredict` is the primary entry point and
`StructureDiffusionInference` wraps it with the noising step, matching upstream.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..diffusion.forward import diffuse_atom_cloud, diffuse_coordinates_edm, diffuse_sequence
from .condition import PrepareCondition
from .diffusion import Diffusion

__all__ = ["StructureDiffusionPredict", "StructureDiffusionInference",
           "apply_diffusion"]


def apply_diffusion(config, data: dict, *, generator: torch.Generator | None = None,
                    noise: Tensor | None = None,
                    seq_noise: Tensor | None = None) -> dict:
    """The forward noising step (`StructureDiffusionNoise`). No parameters.

    `noise` / `seq_noise` let a caller inject pre-drawn randomness so a run can
    be reproduced exactly; that is how the port is checked against JAX.
    """
    c = config
    pos = data["pos"]
    t_pos, t_seq = data["t_pos"], data["t_seq"]
    batch, mask = data["batch_index"], data["mask"]

    if c.diffusion_kind == "edm":
        pos_noised = diffuse_coordinates_edm(pos, batch, t_pos,
                                             generator=generator, noise=noise)
    else:
        # "vpfixed" pins the cloud std at sigma_data; "vp" uses the per-structure
        # CA spread (or an explicit --cloud_std).
        cloud_std = c.sigma_data if c.diffusion_kind == "vpfixed" else data.get("cloud_std")
        pos_noised = diffuse_atom_cloud(
            pos, mask, batch, t_pos, cloud_std=cloud_std,
            flow=c.diffusion_kind in ("flow", "flow_scaled"),
            generator=generator, noise=noise)

    seq_noised, corrupt_aa = diffuse_sequence(data["seq"], t_seq,
                                              generator=generator, noise=seq_noise)
    return {"pos_noised": pos_noised, "seq_noised": seq_noised,
            "t_pos": t_pos, "t_seq": t_seq, "corrupt_aa": corrupt_aa}


class StructureDiffusionPredict(nn.Module):
    """Denoise only: expects ``pos_noised``/``seq_noised`` already in `data`."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.prepare_condition = PrepareCondition(config)
        self.diffusion = Diffusion(config)

    def forward(self, data: dict, prev: dict, *, predict_aa: bool = True,
                uniform: Tensor | None = None,
                compute_distogram: bool = True) -> tuple[dict, dict]:
        data = dict(data)
        data.update(self.prepare_condition(data))
        result = self.diffusion(data, prev, predict_aa=predict_aa,
                                uniform=uniform, compute_distogram=compute_distogram)
        new_prev = {"pos": result["pos"], "local": result["local"]}
        return result, new_prev


class StructureDiffusionInference(nn.Module):
    """Noise, then denoise - the single-call form used by the eval script."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.predict = StructureDiffusionPredict(config)

    @property
    def diffusion(self):
        return self.predict.diffusion

    @property
    def prepare_condition(self):
        return self.predict.prepare_condition

    def forward(self, data: dict, prev: dict, *, predict_aa: bool = True,
                generator: torch.Generator | None = None,
                noise: Tensor | None = None,
                uniform: Tensor | None = None,
                compute_distogram: bool = True) -> tuple[dict, dict]:
        data = dict(data)
        data.update(apply_diffusion(self.config, data,
                                    generator=generator, noise=noise))
        return self.predict(data, prev, predict_aa=predict_aa, uniform=uniform,
                            compute_distogram=compute_distogram)
