# TODO

## Next
- T1 (N003, blocker): fix `modules/encoder.py:302,372` -> `frames.unsqueeze_batch(1)`. The Encoder cannot run at all, and `training/loop.py` now calls it.
- T2 (N003): promote `results/N003-.../r05_encoder_parity_{jax,torch}.py` to `tests/test_encoder_parity.py` so T1 can never silently regress.
- T3 (N003): fix `tests/test_stack_parity.py:93` (`neighbour_fn` needs the `block_idx` arg) and re-verify the 0.0003 A figure in `notes/04_PARITY_AND_PERF.md`.
- T4 (N003): decide the `FastDenoiser` condition cache — key it on the conditioning tensors' identity/version, or drop it. Also fix the static-buffer caller-mutation, `predict_aa=False`, `aa_condition`, and make `compute_distogram=True` raise.
- T5 (N003): `inference/symmetry.py` clash cost -> clamp the difference, not the square; decide whether `c1` and `count | L` should follow JAX or warn.
- T6 (N003): `modules/model.py` `apply_diffusion` -> let `data["cloud_std"]` override `vpfixed`, and add `"flow"` to the `sigma_data` branch; seed `options.py`'s randomness and thread `--seed` to the neighbour draw.

- T7 (N005): `fixtures/*.npz` are gitignored, so the TF32 accident is repeatable by anyone who regenerates them. The generator pin + test guard cover it, but consider checking a small manifest (sha256 + `jax_platform` + `jax_matmul_precision` per fixture) into git so a stale/foreign fixture is visible in `git status`.
- T8 (N005): `tests/test_stack_parity.py` and the encoder/loss fixtures (`fixtures/{stack,block,loss,viol,encoder,geometry,sampling,kabsch}_*.npz`) have NOT been checked for the same TF32 provenance problem - none of them record a backend. Audit how each was generated.

## Unverified claims
- N003 was run by a Claude subagent, not Codex (Codex produced no output). Priors are correlated with the implementer's; a genuine cross-family red team is still owed.
- N003 did not test "batch of two == two separate runs" for `training/` — `loop.py` was being edited concurrently by N004.

## Unresolvable
