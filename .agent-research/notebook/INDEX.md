# Index

| ID | date | type | title | status | outcome | results | commit |
| -- | ---- | ---- | ----- | ------ | ------- | ------- | ------ |
| N001 | 2026-09-25 | task | SALAD JAX to PyTorch port | open |  | .agent-research/results/N001-salad-jax-to-pytorch-port/ |  |
| N002 | 2026-09-25 | task | Generation-loop CLI flags (spec 06): symmetry, template, renoise, order-agnostic | done |  | .agent-research/results/N002-generation-flags/ |  |
| N003 | 2026-09-25 | research | Red-team pass: fast path, symmetry, encoder, generation flags | done | 2 crashes (Encoder dead; stack-parity test broken) + 11 silent-wrongness findings; Encoder math verified exact vs JAX | .agent-research/results/N003-red-team-pass-fast-path-symmetry-encoder/ |  |
| N004 | 2026-09-25 | task | SALAD porting gaps: parity, encoder training wiring, optimizer step | done | Encoder+loss `Rigid` crash fixed (5 sites); Encoder jointly trains (overfit loss falls, aa 2.996->2.287/20 steps); training-step gradients match JAX (cosine 0.9999986); reveal_conditioning transcribed exactly; default_vp_scaled/multimotif_vp parity measured but unresolved (~300-1800x default_vp) | .agent-research/results/N004-porting-gaps/ |  |
