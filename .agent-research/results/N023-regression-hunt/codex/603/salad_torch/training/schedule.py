"""Learning-rate schedules (spec `04_data_and_training.md` §B.3).

Both return a plain `int -> float` giving the ABSOLUTE learning rate for that
step count (not a multiplier), matching upstream's `optax` schedule closely
enough that `training/loop.py` just does `param_group['lr'] = schedule(step)`
every step rather than routing through `LambdaLR`'s "multiplier of base_lr"
convention.
"""
from __future__ import annotations

import math
from typing import Callable

__all__ = ["cosine_decay_schedule", "warmup_schedule"]


def cosine_decay_schedule(start_lr: float, decay_lr: float, warmup_steps: int,
                          decay_steps: int) -> Callable[[int], float]:
    """`default_vp`'s actual schedule. `train_noise_schedule_benchmark.py:68-82`.

    Linear warmup to `start_lr` over `warmup_steps` (`count == warmup_steps`
    already lands exactly on `start_lr`, so the following step's decay branch
    starts continuously from there), then one half-cosine decay to `decay_lr`
    over the next `decay_steps`.

    **Deliberate quirk, reproduced**: the `% decay_steps` means training past
    `warmup_steps + decay_steps` RESTARTS the cosine cycle from `start_lr`
    rather than holding at `decay_lr` forever - a warm-restart schedule with
    an unbounded restart count, inert only if a run stops at or before
    `max_steps = warmup_steps + decay_steps + 1`. Do not clamp this away.
    """
    def schedule(count: int) -> float:
        if count <= warmup_steps:
            result = count / warmup_steps * start_lr
        else:
            phase = (count - warmup_steps) % decay_steps
            result = ((start_lr - decay_lr) * 0.5
                     * (1 + math.cos(math.pi * phase / decay_steps)) + decay_lr)
        return max(result, 0.0)
    return schedule


def warmup_schedule(start_lr: float, warmup_steps: int) -> Callable[[int], float]:
    """`--schedule_fixed True` alternate: linear warmup, then flat forever.

    Not `default_vp`'s actual schedule (that's `cosine_decay_schedule`), but
    trivial to support and named explicitly in spec §B.3 as "port both".
    """
    def schedule(count: int) -> float:
        return min(count / warmup_steps * start_lr, start_lr)
    return schedule
