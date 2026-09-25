"""Compiled / CUDA-graph fast path for inference.

Why this file exists
--------------------
The denoiser is **launch-bound**, not compute-bound. Measured on an H100 at
L=100: one `DiffusionBlock` costs ~8 ms while JAX's *entire* forward is ~3.1 ms,
and the eager forward takes the same 75 ms whether L is 50 or 800. The work is
tiny; the ~1500 kernel launches per step are the cost.

So the fix is not a better algorithm, it is fewer launches: `torch.compile` to
fuse, and CUDA graphs to replay the whole step as one submission. That needs

* a **tensor-in / tensor-out** region - Dynamo does not faithfully trace the
  python-dict shuffling the ergonomic API does (it drops the `condition` key),
  hence `Diffusion.forward_core`;
* **static input buffers** - CUDA graphs replay against fixed addresses, so the
  caller must copy into the same tensors each step rather than pass new ones;
* **cloned outputs** - a graph's output buffers are overwritten by the next
  replay, which is exactly the "accessing tensor output of CUDAGraphs that has
  been overwritten" error an earlier attempt hit.

Compilation is a one-time cost (tens of seconds); the per-design cost is what
this is for.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .diffusion import uniform_draw_count

__all__ = ["FastDenoiser"]

#: the tensor arguments of `Diffusion.forward_core`, in order
_CORE_ARGS = ("pos_noised", "prev_pos", "prev_local", "condition", "dmap", "omap",
              "dmap_mask", "flags", "t_pos", "t_seq", "seq_noised", "aa_gt",
              "pos_gt", "resi", "chain", "batch", "mask", "cyclic_mask", "uniform")

_OUT_NAMES = ("pos", "local", "trajectory", "aa", "aa_features", "dssp",
              "distogram_neighbours", "raw_angles", "angles", "angle_pos", "aatype")


class FastDenoiser(nn.Module):
    """Wraps a `StructureDiffusionPredict` with a compiled, graph-captured core.

    The signature deliberately mirrors the eager model so it can be swapped in:

        fast = FastDenoiser(model, length=100, device="cuda").warmup()
        result, prev = fast(data, prev)

    `mode` is passed to `torch.compile`. "reduce-overhead" enables CUDA graphs
    and is the setting that matters here; "default" fuses but still launches
    per-kernel.
    """

    def __init__(self, model, length: int, device="cuda",
                 mode: str = "reduce-overhead", dtype=torch.float32,
                 capture: bool = True):
        super().__init__()
        self.model = model
        self.config = model.config
        self.length = length
        self.device = torch.device(device)
        self.dtype = dtype
        self.capture = capture and self.device.type == "cuda"
        self.n_draws = uniform_draw_count(model.config)
        self._compiled = torch.compile(model.diffusion.forward_core,
                                       mode=mode, dynamic=False)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_in: dict[str, Tensor] = {}
        self._cond_cache = None
        self._static_out: tuple[Tensor, ...] | None = None

    # ------------------------------------------------------------------
    def _blank_inputs(self) -> dict[str, Tensor]:
        L = self.length
        A = self.model.diffusion.atom_count
        ls = self.config.local_size
        d, f = self.device, self.dtype
        return dict(
            pos_noised=torch.zeros(L, A, 3, device=d, dtype=f),
            prev_pos=torch.zeros(L, A, 3, device=d, dtype=f),
            prev_local=torch.zeros(L, ls, device=d, dtype=f),
            condition=torch.zeros(L, ls, device=d, dtype=f),
            dmap=torch.zeros(L, L, device=d, dtype=f),
            omap=torch.zeros(L, L, 9, device=d, dtype=f),
            dmap_mask=torch.zeros(L, L, device=d, dtype=torch.bool),
            flags=torch.zeros(L, L, 3, device=d, dtype=f),
            t_pos=torch.full((L,), 0.5, device=d, dtype=f),
            t_seq=torch.ones(L, device=d, dtype=f),
            seq_noised=torch.full((L,), 20, device=d, dtype=torch.long),
            aa_gt=torch.full((L,), 20, device=d, dtype=torch.long),
            pos_gt=torch.zeros(L, A, 3, device=d, dtype=f),
            resi=torch.arange(L, device=d),
            chain=torch.zeros(L, device=d, dtype=torch.long),
            batch=torch.zeros(L, device=d, dtype=torch.long),
            mask=torch.ones(L, device=d, dtype=torch.bool),
            cyclic_mask=None,
            # every Gumbel draw, pre-drawn: CUDA graphs cannot capture an RNG op
            uniform=torch.rand(self.n_draws, L, L, device=d, dtype=f)
                         .clamp_(1e-6, 1 - 1e-6),
        )

    def warmup(self, iters: int = 3) -> "FastDenoiser":
        """Compile and, on CUDA, capture the graph. One-time cost."""
        self._static_in = self._blank_inputs()
        args = [self._static_in[k] for k in _CORE_ARGS]
        with torch.no_grad():
            for _ in range(iters):
                out = self._compiled(*args)
            if self.capture:
                # NOTE: do NOT wrap an explicit torch.cuda.CUDAGraph around a
                # mode="reduce-overhead" callable. Inductor already manages
                # cudagraphs internally, and capturing on top of it raises
                # "Offset increment outside graph capture" from the RNG
                # bookkeeping. Extra warmup iterations are all that is needed.
                for _ in range(2):
                    out = self._compiled(*args)
                torch.cuda.synchronize()
        return self

    # ------------------------------------------------------------------
    def _run(self, inputs: dict[str, Tensor]) -> tuple[Tensor, ...]:
        if self.capture:
            # tells inductor a new iteration has begun, so it may reuse the
            # cudagraph's buffers; without it, holding on to a previous output
            # raises "accessing tensor output of CUDAGraphs that has been
            # overwritten by a subsequent run"
            torch.compiler.cudagraph_mark_step_begin()
        if self._static_in:
            # write into persistent buffers so the input ADDRESSES are stable.
            # Fresh tensors each call show up as
            # `cudagraph_recorded_non_static_inputs` and force a copy per input.
            for k in _CORE_ARGS:
                src, dst = inputs.get(k), self._static_in.get(k)
                if src is not None and dst is not None and dst.shape == src.shape:
                    dst.copy_(src)
                elif src is not None:
                    self._static_in[k] = src
            args = [self._static_in.get(k) for k in _CORE_ARGS]
        else:
            args = [inputs.get(k) for k in _CORE_ARGS]
        out = self._compiled(*args)
        # clone for the same reason: these buffers are reused next call
        return tuple(o.clone() if torch.is_tensor(o) else o for o in out)

    def invalidate_condition_cache(self) -> None:
        """Drop the cached conditioning; call if `data`'s condition keys change."""
        self._cond_cache = None

    def forward(self, data: dict, prev: dict, *, predict_aa: bool = True,
                uniform: Tensor | None = None,
                compute_distogram: bool = False,
                generator: torch.Generator | None = None) -> tuple[dict, dict]:
        if uniform is None:
            # drawn OUTSIDE the graph, then copied into the static buffer
            uniform = torch.rand(self.n_draws, self.length, self.length,
                                 device=self.device, dtype=self.dtype,
                                 generator=generator).clamp_(1e-6, 1 - 1e-6)
        # prepare_condition is LOOP-INVARIANT: it depends only on the
        # conditioning inputs and the chain/batch layout, none of which change
        # across the 500 denoising steps of one design. Running it every step
        # cost more than the whole compiled model. Cached per design; call
        # invalidate_condition_cache() if the conditioning changes.
        cond = getattr(self, "_cond_cache", None)
        if cond is None:
            cond = self.model.prepare_condition(data)
            self._cond_cache = cond
        pc = cond["pair_condition"]
        inputs = dict(
            pos_noised=data["pos_noised"], prev_pos=prev["pos"],
            prev_local=prev["local"], condition=cond["condition"],
            dmap=pc["dmap"], omap=pc["omap"],
            dmap_mask=pc["dmap_mask"], flags=pc["flags"],
            t_pos=data["t_pos"], t_seq=data["t_seq"],
            seq_noised=data["seq_noised"], aa_gt=data["aa_gt"],
            pos_gt=data["pos"], resi=data["residue_index"],
            chain=data["chain_index"], batch=data["batch_index"], mask=data["mask"],
            cyclic_mask=data.get("cyclic_mask") if self.config.cyclic else None,
            uniform=uniform,
        )
        out = self._run(inputs)
        result = dict(zip(_OUT_NAMES, out))
        result["atom_pos"] = result.pop("angle_pos")
        result["pos_noised"] = data["pos_noised"]
        result["aa_gt"] = data["aa_gt"]
        if not predict_aa:
            result["aatype"] = data["aa_gt"]
        return result, {"pos": result["pos"], "local": result["local"]}
