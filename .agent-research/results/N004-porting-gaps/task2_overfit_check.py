"""Task 2 self-check: does the training loop still train once prepare_batch
uses a real Encoder (instead of the CB-replication placeholder)?

Draws ONE fixed synthetic batch, then runs `training_step` on it repeatedly
(fresh noise/self-conditioning draw each step, same structure) and checks the
total loss falls - the standard "overfit one example" sanity check for a
diffusion trainer. Also checks the Encoder actually receives nonzero gradient
(the whole point of wiring it in with gradients enabled).

Runs on CPU, tiny synthetic proteins, no checkpoint needed (random init).
"""
import os
import sys

ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)

import torch

from salad_torch.config import get_config
from salad_torch.modules.model import StructureDiffusionPredict
from salad_torch.training.loop import TrainConfig, Trainer, training_step
from salad_torch.data.dataset import SyntheticStructureSource, PackedProteinStream
from salad_torch.data.collate import example_to_tensors

torch.manual_seed(0)

config = get_config("default_vp")
model = StructureDiffusionPredict(config)

train_config = TrainConfig(num_aa=24, min_size=12, max_size=24,
                           lr=1e-3, warmup_steps=1, decay_steps=1000)
source = SyntheticStructureSource(num_clusters=8, chain_length_range=(12, 24))
stream = PackedProteinStream(source, size=train_config.num_aa,
                             min_size=train_config.min_size,
                             max_size=train_config.max_size,
                             p_complex=0.0, legacy_repetitive_chains=True, seed=0)

trainer = Trainer(model, train_config, iter(stream), device="cpu")
assert trainer.encoder is not None, "default_vp has augment_size=15 > 0; Encoder must be built"

# one FIXED draw, reused every step - the actual overfit condition
raw = next(trainer.batches)
fixed_batch = [example_to_tensors(raw, device=trainer.device)]
print(f"fixed batch: {fixed_batch[0]['aa_gt'].shape[0]} residues, "
     f"{int(fixed_batch[0]['mask'].sum())} real")

encoder_params_before = [p.detach().clone() for p in trainer.encoder.parameters()]

gen = torch.Generator(device="cpu").manual_seed(123)
losses = []
for step in range(20):
    lr = trainer.schedule(step)
    terms = training_step(trainer.model, trainer.encoder, trainer.ema, trainer.optimizer,
                          trainer.config, fixed_batch, device=trainer.device,
                          generator=gen, lr=lr, encoder_ema=trainer.encoder_ema)
    losses.append(terms["total"])
    print(f"step {step:2d}  total={terms['total']:.4f}  "
         f"pos={terms.get('pos', float('nan')):.4f}  "
         f"aa={terms.get('aa', float('nan')):.4f}")

print()
print("losses:", [f"{l:.4f}" for l in losses])
print(f"first={losses[0]:.4f} last={losses[-1]:.4f}")

# Encoder must have actually moved (real gradient, real optimizer step)
moved = any(not torch.allclose(a, b) for a, b in
           zip(encoder_params_before, trainer.encoder.parameters()))
print(f"encoder parameters changed after 20 steps: {moved}")

any_grad = any(p.grad is not None for p in trainer.encoder.parameters())
print(f"encoder has grad != None on last step: {any_grad}")

ok = losses[-1] < losses[0] and moved and any_grad
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
