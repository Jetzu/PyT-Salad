"""Inference fast-path switches (N009, executing the N007 PLAN).

Every switch here defaults to **on** and is either bit-exact or
reassociation-only; the environment variables exist so one benchmark job can
measure each proposal separately in its own process. They are read **once, at
import time**, so they are ordinary Python constants by the time Dynamo traces
anything - a mutable flag would become a guard and force a recompile.

| flag | proposal | class |
|---|---|---|
| ``SALAD_OPT_GATHER=0`` | P3 one-hot x matrix -> row gather | bit-exact |
| ``SALAD_OPT_HOIST=0``  | P6 hoist loop-invariant neighbour stages | bit-exact |
| ``SALAD_OPT_NOPC=0``   | P4+P5 skip the dead pair-conditioning work | P4 bit-exact, P5 reassociation |
| ``SALAD_OPT_LEAN=0``   | P2 skip the AA decoder + angle head off-step | skip (see below) |

P4/P5 additionally carry a **runtime** gate: they only fire when the design has
no pair conditioning at all (``dmap_mask`` all false AND ``flags`` all false),
which `FastDenoiser` decides per design from the real conditioning tensors.
The switch below can only ever turn that off, never on.
"""
from __future__ import annotations

import os

__all__ = ["GATHER_ONEHOT", "HOIST_NEIGHBOURS", "SKIP_PAIR_COND", "LEAN_STEPS",
           "NEED_AUX"]


def _flag(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


#: P3 - `sequence_relative_position(one_hot=True) @ W` and `one_hot(aa) @ W`
#: become `W[idx]`. Bit-exact: 65 of the 66 products are exactly 0.0 and
#: `a + 0.0 == a` in IEEE-754 for any finite a, in any summation order.
GATHER_ONEHOT = _flag("SALAD_OPT_GATHER")

#: P6 - compute the sequence-index neighbour stage once instead of 12 times,
#: the self-conditioning branch's distance matrix + index + spatial stages once
#: instead of 6 times, and share the CA distance matrix between the spatial and
#: random stages. Pure common-subexpression elimination, bit-exact.
HOIST_NEIGHBOURS = _flag("SALAD_OPT_HOIST")

#: P4 + P5 - with no pair conditioning the `dmap`/`flags` pair terms are exactly
#: zero and the fourth ("block contact") neighbour stage returns all -1, so the
#: condition branch runs at K=48 instead of K=64. P4 is bit-exact; P5 changes
#: only the reduction tree over K (the dropped slots contribute exactly 0.0).
SKIP_PAIR_COND = _flag("SALAD_OPT_NOPC")

#: P2 - let the sampler ask for the sequence heads only on reporting steps.
#: A skip, not a reassociation: `pos`/`local`/`trajectory` come out of the same
#: trunk. The one thing it does NOT control is whether inductor compiles the
#: two variants' shared trunk into the same kernels, which is why the gate for
#: shipping it is a byte-identical PDB from a full run, not a tensor diff.
LEAN_STEPS = _flag("SALAD_OPT_LEAN")

#: The dssp head and the distogram neighbour draw, neither of which sampling
#: reads. Default ON - removing them was measured SLOWER once (6.48 -> 6.9 ms),
#: back when the forward was 30 broken Dynamo graphs. `SALAD_OPT_AUX=0`
#: re-tests it against the single-graph forward.
NEED_AUX = _flag("SALAD_OPT_AUX")
