# Index

| ID | date | type | title | status | outcome | results | commit |
| -- | ---- | ---- | ----- | ------ | ------- | ------- | ------ |
| N001 | 2026-09-25 | task | SALAD JAX to PyTorch port | open |  | .agent-research/results/N001-salad-jax-to-pytorch-port/ |  |
| N002 | 2026-09-25 | task | Generation-loop CLI flags (spec 06): symmetry, template, renoise, order-agnostic | done |  | .agent-research/results/N002-generation-flags/ |  |
| N003 | 2026-09-25 | research | Red-team pass: fast path, symmetry, encoder, generation flags | done | 2 crashes (Encoder dead; stack-parity test broken) + 11 silent-wrongness findings; Encoder math verified exact vs JAX | .agent-research/results/N003-red-team-pass-fast-path-symmetry-encoder/ |  |
| N004 | 2026-09-25 | task | SALAD porting gaps: parity, encoder training wiring, optimizer step | done | Encoder+loss `Rigid` crash fixed (5 sites); Encoder jointly trains (overfit loss falls, aa 2.996->2.287/20 steps); training-step gradients match JAX (cosine 0.9999986); reveal_conditioning transcribed exactly; default_vp_scaled/multimotif_vp parity measured but unresolved (~300-1800x default_vp) | .agent-research/results/N004-porting-gaps/ |  |
| N005 | 2026-09-25 | fix | Parity bisect: default_vp_scaled and multimotif_vp fixtures were TF32 | done | NOT a port bug: both fixtures were generated on an H100 where XLA lowers fp32 dot to TF32 (`gen_fixtures_task1.sbatch`); regenerating default_vp itself on GPU reproduces the whole failure (out_pos 5.98e-03, traj0 1.22e-02, aatype still exact). Torch matches CPU-JAX to 9.27e-06 at every block-0 tensor. Fixed: precision pin + provenance in the generator, guard in the test; all four configs now 0.00002/0.00020 A | .agent-research/results/N005-parity-bisect/ | 081dd2d |
