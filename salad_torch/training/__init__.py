"""Training loop for the `default_vp` torch port: optimizer, LR schedule, EMA,
checkpointing, and the CLI entry point (`training/train.py`).

Replaces upstream's `flexloop` + `optax` stack (spec
`notes/specs/04_data_and_training.md` Part B). See `training/loop.py`'s module
docstring for the two flagged gaps (the un-ported `Encoder` network, and the
best-effort conditioning-reveal reconstruction) before using this for a real
training run.
"""
from .ema import EMA
from .loop import Trainer, TrainConfig, prepare_batch, reveal_conditioning, training_step
from .schedule import cosine_decay_schedule, warmup_schedule

__all__ = ["EMA", "Trainer", "TrainConfig", "prepare_batch", "reveal_conditioning",
          "training_step", "cosine_decay_schedule", "warmup_schedule"]
