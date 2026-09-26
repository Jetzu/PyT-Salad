"""Base layers, deliberately mirroring Haiku's parameter layout.

Design note: `HkLinear` stores its weight as ``w`` with shape ``(in, out)`` and
computes ``x @ w + b``, i.e. Haiku's layout rather than ``nn.Linear``'s
``(out, in)``. That is a conscious choice - it makes checkpoint conversion a
pure name mapping with no transposes, which removes an entire class of silent
bugs. These are still ordinary `nn.Parameter`s, so training works normally.

Bias quirk reproduced from upstream: `Linear` allocates ``b`` even when
``bias=False``, so roughly 40% of the ``b`` entries in a salad checkpoint are
never added. We allocate it too (so state dicts line up) but do not apply it.
"""
from __future__ import annotations

import math
from functools import partial
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from .. import opt

__all__ = ["HkLinear", "HkLayerNorm", "MLP", "init_weight_", "gelu", "relu"]

# CRITICAL: jax.nn.gelu is `def gelu(x, approximate: bool = True)`, i.e. the TANH
# approximation, and every salad call site takes that default. torch's F.gelu
# defaults to approximate='none' (exact erf). Using the erf variant costs ~0.3 A
# max coordinate deviation over the 6-layer trunk in a single denoiser call, and
# the sampler runs 400-500 of them. Do not "simplify" this back to F.gelu.
gelu = partial(nn.functional.gelu, approximate="tanh")
relu = nn.functional.relu

# Haiku's truncated_normal VarianceScaling divides by the stddev of a standard
# truncated normal on [-2, 2] so the result has the requested variance.
_TRUNC_NORM_STD = 0.87962566103423978


def init_weight_(w: Tensor, scheme: str = "linear") -> None:
    """In-place init matching `salad.modules.basic` initialisers.

    - ``linear``: VarianceScaling(fan_in, scale=1.0), truncated normal
    - ``relu``:   VarianceScaling(fan_in, scale=2.0), truncated normal
    - ``glorot``: VarianceScaling(fan_avg, scale=1.0), *uniform*
    - ``zeros``:  constant 0
    """
    if scheme == "zeros":
        nn.init.zeros_(w)
        return
    fan_in, fan_out = w.shape[0], w.shape[1]
    if scheme == "glorot":
        limit = math.sqrt(3.0 * 1.0 / ((fan_in + fan_out) / 2.0))
        nn.init.uniform_(w, -limit, limit)
        return
    scale = 2.0 if scheme == "relu" else 1.0
    std = math.sqrt(scale / fan_in) / _TRUNC_NORM_STD
    nn.init.trunc_normal_(w, mean=0.0, std=std, a=-2 * std, b=2 * std)


class HkLinear(nn.Module):
    """``y = x @ w + b`` with Haiku's ``(in, out)`` weight layout."""

    def __init__(self, in_size: int, out_size: int, bias: bool = True,
                 bias_init: float = 0.0, initializer: str = "linear"):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.use_bias = bias
        self.w = nn.Parameter(torch.empty(in_size, out_size))
        # allocated unconditionally, applied only when use_bias - see module docstring
        self.b = nn.Parameter(torch.full((out_size,), float(bias_init)))
        init_weight_(self.w, initializer)

    def forward(self, x: Tensor) -> Tensor:
        # N020. This used to be `torch.einsum("...k,ki->...i", x, self.w)`, which
        # ATen lowers to `bmm` with batch=1 - there was not a single `mm`/`addmm`
        # in the whole generated module (N017 section 7). That mattered for three
        # reasons, all measured: a batch-1 `bmm` is a cuBLAS launch at ~5.98 us
        # for problems whose arithmetic is under 1 us; `bmm` forecloses
        # inductor's own matmul TEMPLATES, which N016 measured 1.21-1.37x faster
        # than cuBLAS on this model's narrow projections and BIT-IDENTICAL at
        # fp32; and a template is autotuned PER DEVICE at compile time, which is
        # the adaptive behaviour a consumer-GPU/Blackwell target wants and a
        # fixed cuBLAS heuristic is not.
        #
        # Folding the leading axes into M and calling `mm` is exactly what
        # `at::matmul` does for (>=3-D) @ (2-D) anyway; doing it here makes the
        # lowering explicit instead of depending on ATen's `should_fold`
        # heuristic (which declines for some strides and silently goes back to
        # `bmm`). The weight keeps Haiku's `(in, out)` layout - no transpose, so
        # checkpoint conversion stays a pure name mapping.
        mode = opt.MM_MODE
        if mode == "einsum":
            out = torch.einsum("...k,ki->...i", x, self.w)
            if self.use_bias:
                out = out + self.b
            return out
        ndim = x.dim()
        flat = x if ndim == 2 else x.reshape(-1, x.shape[-1])
        if mode == "addmm" and self.use_bias:
            out = torch.addmm(self.b, flat, self.w)
        else:
            out = torch.mm(flat, self.w)
            if self.use_bias:
                out = out + self.b
        if ndim != 2:
            out = out.view(*x.shape[:-1], self.out_size)
        return out

    def forward_indexed(self, idx: Tensor) -> Tensor:
        """`forward(one_hot(idx, in_size))` written as the row gather it is.

        **Bit-exact**, not approximately: row `r` of `one_hot @ w` is
        `sum_k onehot[k] * w[k, :]`, in which `in_size - 1` terms are exactly
        `0.0 * finite = 0.0` and one is `w[idx, :]`. IEEE-754 guarantees
        `a + 0.0 == a` for finite `a` in any order, so no summation-order
        argument is needed - the two results are the same bits (the only
        difference a conforming implementation may produce is the sign of a
        zero, which nothing downstream distinguishes).

        The caller owns the range check: `idx` must be in `[0, in_size)`.
        `jax.nn.one_hot` returns an all-zero row for an out-of-range index and
        this does not - it faults (loudly) instead. Only use it where the bin
        derivation makes the range structural.
        """
        out = self.w[idx]
        if self.use_bias:
            out = out + self.b
        return out

    def extra_repr(self) -> str:  # pragma: no cover
        return f"{self.in_size}->{self.out_size}, bias={self.use_bias}"


class HkLayerNorm(nn.Module):
    """LayerNorm over the last axis with Haiku's ``scale``/``offset`` names."""

    def __init__(self, size: int, create_scale: bool = True,
                 create_offset: bool = True, eps: float = 1e-5):
        super().__init__()
        self.size = size
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(size)) if create_scale else None
        self.offset = nn.Parameter(torch.zeros(size)) if create_offset else None

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) * torch.rsqrt(var + self.eps)
        if self.scale is not None:
            out = out * self.scale
        if self.offset is not None:
            out = out + self.offset
        return out


class MLP(nn.Module):
    """`salad.modules.basic.MLP`: depth-1 hidden layers then a final projection."""

    def __init__(self, in_size: int, size: int = 64, out_size: int | None = None,
                 depth: int = 2, activation: Callable[[Tensor], Tensor] = relu,
                 bias: bool = True, init: str = "relu", final_init: str = "linear"):
        super().__init__()
        out_size = out_size if out_size is not None else size
        self.activation = activation
        layers = []
        cur = in_size
        for idx in range(depth):
            if idx < depth - 1:
                layers.append(HkLinear(cur, size, bias=bias, initializer=init))
                cur = size
            else:
                layers.append(HkLinear(cur, out_size, bias=bias, initializer=final_init))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        n = len(self.layers)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < n - 1:
                x = self.activation(x)
        return x
