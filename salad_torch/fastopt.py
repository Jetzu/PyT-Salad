"""Switches for the **fast** inference preset (N011).

This is the permissive twin of `salad_torch/opt.py`. Everything in `opt.py` is
bit-exact or reassociation-only and is on by default, so the port keeps its
0.00002 A parity with JAX salad. Everything *here* changes the numbers on
purpose and is therefore **off by default**: it is turned on only by
``FastDenoiser(..., preset="fast")``, which owns the decision.

Each switch is a module-level Python bool read inside the compiled region.
Dynamo guards on its value, so it must be set *before* `warmup()` and left
alone afterwards - flipping one later forces a recompile and, under a captured
CUDA graph, would be ignored outright because the graph is already recorded.
`configure()` is the only supported way to set them.

Measured dead ends kept OUT of this module (N011, L=100, H100, whole-forward
CUDA graph, against the exact preset's 3.393 ms lean step):

* `torch.topk` in place of the stable argsort in neighbour selection:
  **3.582 ms, i.e. SLOWER by 5.6%.** topk materialises values as well as
  indices and inductor fuses the argsort path better; there is nothing to win.
* `need_aux=False` (drop the dssp head and the distogram neighbour draw):
  3.416 ms, no change. Confirms the same measurement under the older
  inductor-managed graph (notes/04, 6.48 -> 6.9 ms).
* fp16 / bf16 for the whole graph: both abort before running - `at::Half`
  overflow on the `+inf` sentinels and a Float/BFloat16 mismatch. Not chased:
  with TF32 doing the GEMMs the remainder is launch-bound, not
  bandwidth-bound, so the ceiling is the ~0.37 ms memory floor of a 2.8 ms
  step.
* Reusing the current branch's neighbour list across blocks ("stride"):
  crashes Dynamo (``InstructionTranslator has no attribute 'parent'``) when
  written as a stateful closure, and the predicted gain was ~0.1 ms against a
  large structural deviation. Dropped.
"""
from __future__ import annotations

__all__ = ["SHARED_COND_DRAW", "configure", "state"]

#: One Gumbel draw for the self-conditioning branch, shared by all six blocks.
#:
#: After the P6 hoist the condition branch's distance matrix, sequence-index
#: stage and spatial stage are already shared between blocks; only the random
#: stage's draw differs. Pinning that draw makes `cond_neighbours` a single
#: tensor for the whole trunk - and then every geometric feature `pair_2`
#: builds (distance RBF, direction, rotation, pair vectors, relative position)
#: is the same expression on the same inputs in all six blocks, so the compiler
#: builds them once instead of six times. Only the per-block weight matrices
#: still differ.
#:
#: That is P7's observation - "the condition branch is a prologue" - obtained
#: without P7's bmm refactor, and paid for by six blocks sampling the *same*
#: 16 random self-conditioning neighbours instead of six independent sets.
SHARED_COND_DRAW = False


def configure(shared_cond_draw: bool | None = None) -> None:
    """Set the switches. Call before `FastDenoiser.warmup()`, never after."""
    global SHARED_COND_DRAW
    if shared_cond_draw is not None:
        SHARED_COND_DRAW = bool(shared_cond_draw)


def state() -> dict:
    """What is on, for logging next to a benchmark number."""
    return dict(shared_cond_draw=SHARED_COND_DRAW)
