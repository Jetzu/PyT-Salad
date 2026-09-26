# TODO

## Next
- T1 (N003, blocker): DONE in N004 — `modules/encoder.py:302,372` now use `frames.unsqueeze_batch(1)` (read 2026-09-26; see KNOWLEDGE K001/K009).
- T2 (N003): DONE — `tests/test_encoder_parity.py` exists (read 2026-09-26, not re-run to confirm it passes).
- T3 (N003): DONE — `tests/test_stack_parity.py:93`'s `neighbour_fn` already takes `block_idx` (read 2026-09-26). The 0.0003 A figure in `notes/04_PARITY_AND_PERF.md` still wants a re-verify against the fixed test (not done by this lint - reading only, no job run).
- T4 (N003): decide the `FastDenoiser` condition cache — key it on the conditioning tensors' identity/version, or drop it. Also fix the static-buffer caller-mutation, `predict_aa=False`, `aa_condition`, and make `compute_distogram=True` raise.
  - Partially DONE per `salad_torch/modules/fast.py` (read, not re-reproduced by this lint): the cache is now a blake2b content hash (`fast.py:434-450`, explicitly commented "red team N010/S1"), and length mismatches are validated (`fast.py:278,371`, commented "N010/S2"). `compute_distogram=True` is still accepted but does nothing (S8, still open — no raise, no implementation).
- T5 (N003): `inference/symmetry.py` clash cost -> clamp the difference, not the square; decide whether `c1` and `count | L` should follow JAX or warn. Independently reconfirmed by N010/S7 (same `clash_repulsion_grad`/`chain_center` site, up to 6.76e-5 gradient delta) — see KNOWLEDGE K005.
- T6 (N003): `modules/model.py` `apply_diffusion` -> let `data["cloud_std"]` override `vpfixed`, and add `"flow"` to the `sigma_data` branch (DONE per `model.py:44-51`, read 2026-09-26 — see KNOWLEDGE K006); seed `options.py`'s randomness and thread `--seed` to the neighbour draw (still open — `options.py` reads as of 2026-09-26 still uses stdlib `random`, unseeded).

- T7 (N005): `fixtures/*.npz` are gitignored, so the TF32 accident is repeatable by anyone who regenerates them. The generator pin + test guard cover it, but consider checking a small manifest (sha256 + `jax_platform` + `jax_matmul_precision` per fixture) into git so a stale/foreign fixture is visible in `git status`.
- T8 (N005): `tests/test_stack_parity.py` and the encoder/loss fixtures (`fixtures/{stack,block,loss,viol,encoder,geometry,sampling,kabsch}_*.npz`) have NOT been checked for the same TF32 provenance problem - none of them record a backend. Audit how each was generated.

- T9 (N010, unresolved, high): the symmetry hook constrains `snapshot["pos"]` but `snapshot_to_protein`/the PDB writer reads the unconstrained `snapshot["atom_pos"]` (`salad_torch/diffusion/sampling.py:170-188`, `salad_torch/io/pdb.py:172-180`) — written PDBs are not actually symmetric. Upstream has the identical bug (not a new port divergence), so fixing it means deciding whether to diverge from upstream on purpose.
- T10 (N010, unresolved, verification gap): `tests/harness/check_loss.py`/`check_loss2.py` print `FAIL` for a bad term but exit 0; `pytest tests` collects 0 tests despite 5 working `test_*.py` scripts (no `test_`-prefixed functions). Give pytest real test functions and make the loss checkers assert/exit nonzero.
- T11 (N010, unresolved, medium): model-fixture provenance (`backend`/`precision` fields, per K014/T7) is self-asserted and only present on the 4 `model_*.npz` fixtures — a relabelled or perturbed (+0.01 on `out_raw_angles`, inside tolerance) fixture is silently accepted. Overlaps T7/T8's manifest idea; do together.
- T12 (N010, unresolved, low): symmetry leaves unconstrained tails at order>L or the last residue at small L (e.g. L=9,C4), and `c1` skips upstream's `cyclic_centering`. `--sym c0` divides by zero in the CLI parser (crash, C1). The synthetic training data source crashes at the documented minimum `--num-aa 16` (`numpy.random.randint(16,16)` empty interval, C2).
- T13 (N012, unresolved): the "3.4-3.5 ms hard floor" language in N009's outcome is contradicted by N012's from-scratch GEMM-batching measurement (1.87-2.13x speedup on the shipping condition-branch shapes) — re-check N014's batched-GEMM result against N012's K=48-vs-K=64 correction before quoting the older 76%-of-peak/0.21ms figures.
- T14 (integration, process): N010 and N012 are Codex red-team results that were never triaged by a Claude integrator (per the four-role workflow) — they sat with no INDEX row or item file until this lint backfilled them 2026-09-26. T9-T13 above are read-only observations from this lint pass, not a first-hand reproduction; reproduce each before fixing, per the reviewers-report/you-decide rule.
- T15 (N015): designability is the gate no quality work in this port has run - every metric so far is structural (bond geometry, Rg, CA-only P-SEA). ProteinMPNN + a folding model is what `steps_scale=0.5` needs to go from "structurally indistinguishable at n=160" to "free", and it is the precondition for offering 125/100 as a `faster` preset at all.
- T16 (N015): `scripts/generate.py` prints CA-CA per design but never flags a diverged one. A one-line warning on CA-CA mean > 4.5 A or Rg beyond ~3 sd would have surfaced the 2/160 `exact` divergences at run time instead of in a post-hoc script. Out of N015's scope; cheap.

## Unverified claims
- N003 was run by a Claude subagent, not Codex (Codex produced no output). Priors are correlated with the implementer's; a genuine cross-family red team is still owed.
- N003 did not test "batch of two == two separate runs" for `training/` — `loop.py` was being edited concurrently by N004.

## Unresolvable
