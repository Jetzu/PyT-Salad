"""Noise schedules and the CLI expression parsers that select them.

Two things live here that upstream keeps apart:

* the ``sigma(t)`` schedules themselves (`salad.modules.noise_schedule_benchmark`
  for cosine/framediff/linear, `salad.inference.utils.options` for ve/edm), and
* the *parsers* that turn ``--timescale_pos "cosine(t)"`` into a callable.

Upstream implements those parsers with ``eval(data, None, {...})``, which will
happily run ``__import__('os').system(...)`` from a command line or a config
file. We do not. `parse_timescale` and `parse_cloud_std` accept exactly the
documented expression grammar (§1.3 of the spec: named schedule calls with
optional keywords, arithmetic on ``t`` / ``num_aa``, numeric literals, and a
whitelisted ``np.*`` namespace) and reject everything else with a `ValueError`.
The grammar is enforced on the AST, not with a regex, so there is no way to
smuggle an attribute chain or a call through it.

Float precision: JAX runs with x64 disabled, so these functions do their
pre-transcendental arithmetic in Python float64 and then collapse to float32 at
the first ``jnp.*`` call. Reproducing that boundary exactly is worth ~1e-7 of
agreement, which is the difference between "parity" and "close enough, probably".
Each function below marks where the cast happens. `sigma_scale_ve` is the
exception: it is pure Python arithmetic upstream and stays float64 until the
caller broadcasts it over a float32 vector.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "sigma_scale_cosine", "sigma_scale_framediff", "sigma_scale_none",
    "sigma_scale_linear", "sigma_scale_ve", "sigma_scale_edm",
    "cloud_std_default", "SCHEDULE_REGISTRY", "CLOUD_STD_REGISTRY",
    "NP_NAMESPACE", "parse_timescale", "parse_cloud_std", "parse_out_steps",
    "parse_num_aa", "safe_eval",
]

ScalarOrTensor = float | Tensor


def _f32(value: float) -> float:
    """Round a Python float through float32 - the x64-disabled JAX boundary."""
    return float(np.float32(value))


# ---------------------------------------------------------------------------
# sigma(t) schedules
# ---------------------------------------------------------------------------
def sigma_scale_cosine(t: ScalarOrTensor, s: float = 0.01) -> ScalarOrTensor:
    """Cosine schedule. ``noise_schedule_benchmark.py:62-72``.

    ``alpha_bar`` is clipped to ``[0, 1]``; sigma is *not* clipped.
    """
    if torch.is_tensor(t):
        alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / torch.cos(
            torch.as_tensor(s / (1 + s) * math.pi / 2, dtype=t.dtype,
                            device=t.device)) ** 2
        return torch.sqrt(1 - torch.clamp(alpha_bar, 0.0, 1.0))
    arg = (float(t) + s) / (1 + s) * math.pi / 2   # float64, as in JAX
    cos_t = _f32(math.cos(arg))                    # f32 from jnp.cos onward
    alpha_bar = _f32(cos_t * cos_t)
    cos_s = _f32(math.cos(s / (1 + s) * math.pi / 2))
    alpha_bar = _f32(alpha_bar / _f32(cos_s * cos_s))
    alpha_bar = min(max(alpha_bar, 0.0), 1.0)
    return _f32(math.sqrt(_f32(1.0 - alpha_bar)))


def sigma_scale_framediff(t: ScalarOrTensor, bmin: float = 0.1,
                          bmax: float = 20.0) -> ScalarOrTensor:
    """FrameDiff's variance-preserving schedule. ``:75-79``. No clipping."""
    if torch.is_tensor(t):
        gs = t * bmin + 0.5 * t ** 2 * (bmax - bmin)
        return torch.sqrt(1 - torch.exp(-gs))
    gs = float(t) * bmin + 0.5 * float(t) ** 2 * (bmax - bmin)   # float64
    decay = _f32(math.exp(-gs))                                  # f32 from jnp.exp
    return _f32(math.sqrt(_f32(1.0 - decay)))


def sigma_scale_none(t: ScalarOrTensor) -> ScalarOrTensor:
    """Identity schedule, ``sigma = t``. ``:82-84``."""
    return t


#: ``"linear"`` and ``"none"`` are bound to the same function upstream (:87-92).
sigma_scale_linear = sigma_scale_none


def sigma_scale_ve(t: ScalarOrTensor, sigma_max: float = 80.0,
                   rho: float = 7.0) -> ScalarOrTensor:
    """Karras/EDM rho-schedule, increasing in ``t``. ``options.py:79-83``.

    ``sigma_min`` is a hard-coded local upstream, not a parameter, so it stays
    one here too. Returns float64 for scalar input: upstream is pure Python
    arithmetic and only meets float32 when the caller broadcasts it.
    """
    sigma_min = 0.05
    lo, hi = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    if torch.is_tensor(t):
        return (hi + (1 - t) * (lo - hi)) ** rho
    return (hi + (1 - float(t)) * (lo - hi)) ** rho


def sigma_scale_edm(t: ScalarOrTensor) -> ScalarOrTensor:
    """Log-normal EDM schedule. ``options.py:85-90``.

    ``exp(1.6 + 1.4 * Phi^-1(clip(t, 0.001, 0.999)))``. Upstream casts the ppf
    result to float32 *before* the affine and the exp, so the whole tail runs in
    float32 - including the constants, which become ``float32(1.4)``, not 1.4.
    The scalar path therefore stays in a float32 tensor rather than dropping
    into Python floats; doing the affine in float64 costs an ulp.

    Verified against JAX at t = 0, 0.2, 0.5, 0.8: bit-identical. At t >= 0.999
    (the clip) the result is one ulp below XLA's, because XLA's float32 ``expf``
    is one ulp off the correctly-rounded value there and ATen's is not. Nothing
    to fix on this side.
    """
    pos_mean_sigma, pos_std_sigma = 1.6, 1.4
    if not torch.is_tensor(t):
        t = torch.tensor(min(max(float(t), 0.001), 0.999), dtype=torch.float32)
        ppf = torch.special.ndtri(t)
        return float(torch.exp(pos_mean_sigma + pos_std_sigma * ppf))
    ppf = torch.special.ndtri(torch.clamp(t, 0.001, 0.999).to(torch.float32))
    return torch.exp(pos_mean_sigma + pos_std_sigma * ppf)


def cloud_std_default(num_aa: int, *, generator: Any = None,
                      uniform: float | None = None) -> float:
    """``num_aa ** 0.4 + U(0, 3)``. ``options.py:109-112``.

    Upstream draws from numpy's *global* Mersenne Twister, which is not seeded
    by ``--jax_seed`` - so ``--cloud_std "default(num_aa)"`` runs are not
    reproducible from the jax seed alone (quirk 17). Bit-exact parity with a
    JAX run is therefore impossible through the RNG; pass the drawn value in as
    ``uniform=`` instead, which is the only route that actually reproduces.

    `generator` may be a `torch.Generator`, a `numpy.random.Generator`, or
    `None` for the numpy global (upstream behaviour, honours `np.random.seed`).
    """
    minval = num_aa ** 0.4
    if uniform is None:
        if generator is None:
            uniform = float(np.random.rand())
        elif isinstance(generator, torch.Generator):
            uniform = float(torch.rand((), generator=generator,
                                       device=generator.device))
        else:
            uniform = float(generator.random())
    return minval + uniform * 3


SCHEDULE_REGISTRY: dict[str, Callable[..., ScalarOrTensor]] = {
    "cosine": sigma_scale_cosine,
    "framediff": sigma_scale_framediff,
    "none": sigma_scale_none,
    "linear": sigma_scale_linear,
    "ve": sigma_scale_ve,
    "edm": sigma_scale_edm,
}

CLOUD_STD_REGISTRY: dict[str, Callable[..., float]] = {
    "default": cloud_std_default,
}


# ---------------------------------------------------------------------------
# Restricted expression evaluator (the eval() replacement)
# ---------------------------------------------------------------------------
def _dispatch(torch_fn: Callable, py_fn: Callable) -> Callable:
    """Pick the torch or the plain-Python implementation from the arguments."""

    def inner(*args):
        if any(torch.is_tensor(a) for a in args):
            return torch_fn(*[a if torch.is_tensor(a) else torch.as_tensor(a)
                              for a in args])
        return py_fn(*args)

    return inner


def _py_clip(x, lo, hi):
    return min(max(x, lo), hi)


def _py_sign(x):
    return float((x > 0) - (x < 0))


#: The only names reachable through ``np.<attr>``. Explicit whitelist, so
#: dunder traversal is impossible by construction.
NP_NAMESPACE: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "exp": _dispatch(torch.exp, math.exp),
    "log": _dispatch(torch.log, math.log),
    "log10": _dispatch(torch.log10, math.log10),
    "sqrt": _dispatch(torch.sqrt, math.sqrt),
    "abs": _dispatch(torch.abs, abs),
    "sin": _dispatch(torch.sin, math.sin),
    "cos": _dispatch(torch.cos, math.cos),
    "tan": _dispatch(torch.tan, math.tan),
    "tanh": _dispatch(torch.tanh, math.tanh),
    "clip": _dispatch(torch.clamp, _py_clip),
    "minimum": _dispatch(torch.minimum, min),
    "maximum": _dispatch(torch.maximum, max),
    "power": _dispatch(torch.pow, pow),
    "square": _dispatch(torch.square, lambda x: x * x),
    "sign": _dispatch(torch.sign, _py_sign),
    "floor": _dispatch(torch.floor, lambda x: float(math.floor(x))),
    "ceil": _dispatch(torch.ceil, lambda x: float(math.ceil(x))),
}

class _NpProxy:
    """Stand-in for the ``np`` module inside parsed expressions.

    The AST visitor intercepts ``np.<attr>`` and resolves it against
    `NP_NAMESPACE` without ever touching this object, so it exists only to give
    the bare name ``np`` something to bind to.
    """


_BINOPS: dict[type, Callable] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_UNARYOPS: dict[type, Callable] = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}

_MAX_EXPR_LEN = 512
_MAX_DEPTH = 32

#: ``cosine(t)`` and friends - the 95% case, resolved without touching the AST.
_BARE_CALL = re.compile(r"^([a-z_][a-z_0-9]*)\((t|num_aa)\)$")


class _SafeEvaluator(ast.NodeVisitor):
    """Evaluates the restricted grammar; raises `ValueError` on anything else."""

    def __init__(self, names: dict[str, Any]):
        self.names = names

    def visit(self, node, depth: int = 0):  # type: ignore[override]
        if depth > _MAX_DEPTH:
            raise ValueError(f"expression nested deeper than {_MAX_DEPTH} levels")
        handler = getattr(self, "_do_" + type(node).__name__, None)
        if handler is None:
            raise ValueError(
                f"disallowed expression element {type(node).__name__!r}")
        return handler(node, depth)

    def _do_Expression(self, node, depth):
        return self.visit(node.body, depth + 1)

    def _do_Constant(self, node, depth):
        # bool is a subclass of int, so it needs an explicit reject
        if type(node.value) not in (int, float):
            raise ValueError(f"only int/float literals allowed, got {node.value!r}")
        return node.value

    def _do_Name(self, node, depth):
        if node.id not in self.names:
            raise ValueError(f"unknown name {node.id!r}")
        return self.names[node.id]

    def _do_Attribute(self, node, depth):
        if not (isinstance(node.value, ast.Name) and node.value.id == "np"):
            raise ValueError("only 'np.<name>' attribute access is allowed")
        if node.attr not in NP_NAMESPACE:
            raise ValueError(f"'np.{node.attr}' is not whitelisted")
        return NP_NAMESPACE[node.attr]

    def _do_Call(self, node, depth):
        func = self.visit(node.func, depth + 1)
        if not callable(func):
            raise ValueError("attempted to call a non-callable")
        args = [self.visit(a, depth + 1) for a in node.args]
        if any(isinstance(a, ast.Starred) for a in node.args):
            raise ValueError("*args is not allowed")
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError("**kwargs is not allowed")
            kwargs[kw.arg] = self.visit(kw.value, depth + 1)
        return func(*args, **kwargs)

    def _do_BinOp(self, node, depth):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"disallowed operator {type(node.op).__name__!r}")
        return op(self.visit(node.left, depth + 1), self.visit(node.right, depth + 1))

    def _do_UnaryOp(self, node, depth):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"disallowed unary operator {type(node.op).__name__!r}")
        return op(self.visit(node.operand, depth + 1))


def safe_eval(expr: str, names: dict[str, Any]) -> Any:
    """Evaluate `expr` under the restricted grammar. Never calls `eval`."""
    if len(expr) > _MAX_EXPR_LEN:
        raise ValueError(f"expression longer than {_MAX_EXPR_LEN} characters")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse expression {expr!r}: {exc}") from exc
    return _SafeEvaluator(names).visit(tree)


# ---------------------------------------------------------------------------
# CLI parsers
# ---------------------------------------------------------------------------
def parse_timescale(spec: str | Callable) -> Callable[[ScalarOrTensor], ScalarOrTensor]:
    """``--timescale_pos`` / ``--timescale_seq`` -> ``sigma(t)``.

    Accepts an already-callable spec unchanged (mirrors
    `Sampler._get_timescale`). Unlike upstream, the expression is parsed *once*
    and the callable cached - upstream re-parses and re-``eval``s it on every
    one of the 500 steps (quirk 18). The schedules are pure, so this is
    semantically free.
    """
    if callable(spec):
        return spec
    text = spec.strip()

    bare = _BARE_CALL.match(text)
    if bare is not None and bare.group(2) == "t":
        fn = SCHEDULE_REGISTRY.get(bare.group(1))
        if fn is not None:
            return fn

    names = dict(SCHEDULE_REGISTRY)
    names["np"] = _NpProxy

    def inner(t: ScalarOrTensor) -> ScalarOrTensor:
        return safe_eval(text, {**names, "t": t})

    return inner


def parse_cloud_std(spec: str | Callable | None,
                    *, generator: Any = None) -> Callable[[int], float | None]:
    """``--cloud_std`` -> ``f(num_aa) -> float | None``.

    ``"none"`` is short-circuited *before* parsing, exactly as upstream does, and
    maps to `None` (meaning "derive the cloud std from the current positions
    with `position_std`"), not to the float 0.
    """
    if spec is None:
        return lambda num_aa: None
    if callable(spec):
        return spec
    text = spec.strip()
    if text == "none":
        return lambda num_aa: None

    def default(num_aa: int, **kwargs) -> float:
        kwargs.setdefault("generator", generator)
        return cloud_std_default(num_aa, **kwargs)

    names = {"default": default, "np": _NpProxy}

    def inner(num_aa: int) -> float:
        return safe_eval(text, {**names, "num_aa": num_aa})

    return inner


def parse_out_steps(spec: str | Sequence[int] | int) -> list[int]:
    """``--out_steps`` -> list of step indices. ``options.py:72-77``.

    **Not sorted.** The sampler breaks on ``out_steps[-1]``, the last element as
    *written*, so ``"400,100"`` stops the run at step 100 (quirk 2). Sorting
    here would silently change behaviour.
    """
    if isinstance(spec, int):
        return [spec]
    if not isinstance(spec, str):
        return [int(x) for x in spec]
    if spec.strip() == "all":
        return list(range(1000))
    return [int(c) for c in spec.split(",")]


def parse_num_aa(spec: str | int, *, device: torch.device | str | None = None
                 ) -> tuple[int, Tensor, Tensor, bool, Tensor]:
    """``--num_aa`` -> ``(num_aa, residue_index, chain_index, is_cyclic, cyclic_mask)``.

    ``"100:200:c8"`` = three chains of 100, 200 and 8 residues, the last cyclic.

    Residue numbering starts at **0** and each new chain starts 50 past the
    *previous residue's* index - so ``"100:100"`` gives ``0..99`` then
    ``149..248``, not ``150..249`` (quirk 15). Reproduced deliberately; both the
    off-by-one and the 0-based start reach the output PDB.
    """
    resi: list[int] = []
    chain: list[int] = []
    cyclic: list[bool] = []
    for idx, size_str in enumerate(str(spec).split(":")):
        size_str = size_str.strip()
        is_c = size_str.startswith("c")
        size = int(size_str[1:] if is_c else size_str)
        last_resi = resi[-1] + 50 if resi else 0
        resi += [last_resi + i for i in range(size)]
        chain += [idx] * size
        cyclic += [is_c] * size
    num_aa = len(resi)
    resi_t = torch.tensor(resi, dtype=torch.int32, device=device)
    chain_t = torch.tensor(chain, dtype=torch.int32, device=device)
    cyclic_t = torch.tensor(cyclic, dtype=torch.bool, device=device)
    return num_aa, resi_t, chain_t, bool(any(cyclic)), cyclic_t
