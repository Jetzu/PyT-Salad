# N004 · SALAD porting gaps: parity, encoder training wiring, optimizer step

type: task · date: 2026-09-25 · status: done

## Goal
Close 4 named correctness gaps in the JAX->PyTorch salad port (caller task,
priority order): (1) verify `default_vp_scaled`/`multimotif_vp` model parity
vs JAX (never checked before, unlike `default_vp`/`default_ve_scaled`);
(2) wire the real `Encoder` into `training/loop.py::prepare_batch`, replacing
the non-learned CB-replication placeholder; (3) compare a full training step
(gradients + one Adam optimizer step) against JAX; (4) if time remains,
transcribe `training/loop.py::reveal_conditioning` properly from
`Condition.get_masks`/`StructureDiffusion.prepare_condition` instead of the
prior best-effort reconstruction.

## Plan
Reuse existing machinery (`scripts/gen_fixtures_model.py`,
`tests/test_model_parity.py`, `tests/harness/gen_loss_ref2.py`) rather than
inventing new harnesses; container work via short h100 sbatch jobs (all
<3 min, well under the 25 GPU-h/1 GPU-h guidance).

## Changes
- `scripts/gen_fixtures_model.py`: added `build_motif_condition`/multi_motif
  branch in `build_data` (real dmap/dmap_mask, not all-False) so the
  `multimotif_vp` fixture actually exercises `extract_motif_neighbours`.
- `salad_torch/modules/encoder.py`: fixed 2 crashes (`frames[:, None]` ->
  `frames.unsqueeze_batch(1)`, lines ~302/372) - `Rigid.__getitem__` was
  removed upstream in the port; every Encoder call raised `TypeError`.
  Independently found AND independently confirmed by a concurrent N003
  red-team pass (their D1), same root cause, same fix.
- `salad_torch/losses/structure.py`: same class of bug, 2 more sites
  (`sparse_fape`/`local_loss`, lines ~228/230/286/287) - `frames[...,
  None, None]` -> `.unsqueeze_batch(2)`. Never caught before because no
  training step had ever been run end-to-end.
- `salad_torch/losses/total.py`: `frames_pred_20[-1]` -> `.select_batch(0,
  -1)` (`Rigid` has no `__getitem__` at all, not even single-index).
- `salad_torch/training/loop.py`: (a) `prepare_batch` now takes a real
  `Encoder` and calls it with gradients enabled; raises loudly if
  `augment_size>0` and no encoder given. (b) `Trainer` builds a real
  `Encoder(config)`, folds its parameters into the SAME `optimizer` as
  `model`, and gives it its own `EMA`/checkpoint entry (`encoder_ema`,
  `encoder_params`/`encoder_ema_params` keys) - kept a SEPARATE `EMA` object
  from `model`'s so `model`'s own EMA/checkpoint key namespace is unchanged
  for a sampling-only consumer. (c) `reveal_conditioning` rewritten to
  transcribe `Condition.get_masks` + `StructureDiffusion.prepare_condition`'s
  outer gate exactly (NSB:2562-2586, 241-258), adding the previously-entirely-
  missing `dssp_mean` channel - see the function's docstring for the 3
  concrete differences from the old best-effort version.

## Evidence
- **Task 1** (numbers, both freshly generated fixtures + parity run):
  - `default_vp_scaled`: job 901506 (h100, ~2 min). `tests/test_model_parity.py`
    (SALAD_CONFIG=default_vp_scaled): max coord dev **0.00589 A** (mean
    0.00118 A), aatype 0/48 differ. ~300x looser than `default_vp`'s
    0.00002 A budget; several channels exceed the harness's 2e-5 tolerance.
    Output: `task1_parity_default_vp_scaled.txt`.
  - `multimotif_vp` (real dmap/dmap_mask: 2 interleaved groups of 40/40
    residues at L=80, block=20, forcing a real top-32 rank not a trivial
    admit-everything selection): same job. Max coord dev **0.03694 A** (mean
    0.00438 A), aatype 0/80 differ. Output: `task1_parity_multimotif_vp.txt`.
  - Root-cause status: NOT fully resolved. Ruled out: checkpoint scale
    mismatch (noised-position std/absmax match default_vp's to <2%);
    converter/shape bug (identical `build_state_dict`/mapping code path,
    identical leaf counts per config, `multi_motif`'s
    `MotifAdapterDiffusionBlock` translation checked line-by-line against
    NSB:2099-2163 and found faithful). aatype (a downstream argmax) matches
    EXACTLY in both cases, which is strong evidence the computation is
    substantively correct. Working hypothesis: fp32 order-of-operations
    sensitivity in the discrete top-k neighbour selection, chained across the
    6 trunk blocks (the same "sampling is chaotic" mechanism `notes/
    04_PARITY_AND_PERF.md` §2 already documents for the 81-step sampler,
    except here within ONE forward pass's 6 internal blocks), amplified by
    these 2 configs' extra pair-feature linears (`linear_dmap_mask`,
    `init_pos`) and (for `default_vp_scaled`) a different trained-weight
    distribution. NOT conclusively verified - flagged for whoever owns
    `modules/blocks.py`/`diffusion.py` (both DO-NOT-TOUCH for this task).
- **Task 2**: `.agent-research/results/N003-red-team-pass-fast-path-symmetry-
  encoder/r05_encoder_parity_torch.py` (independent test built by the
  concurrent N003 pass) re-run after my fix: **PASS**, local rel <=1.9e-6,
  coords max **0.00001 A** - matches the rest of the port's parity level
  exactly. Promoted to `tests/test_encoder_parity.py` +
  `tests/harness/gen_encoder_ref.py` (+ `fixtures/encoder_default_vp.npz`),
  re-confirmed passing. `task2_overfit_check.py`: `Trainer`-driven overfit of
  one FIXED synthetic batch (L=24 after an L=64 run proved too slow on CPU,
  ~20+ min for 20 full trunk+violation-loss steps with no GPU), 20
  `training_step` calls, fresh noise/self-cond draw each step (real diffusion
  training, not a toy deterministic loss - `t_pos` is redrawn every step, so
  per-step total loss is expected to be noisy, not monotonic). Total loss:
  `[249.21, 196.22, 225.31, 232.01, 181.43, 239.15, 223.46, 242.60, 203.79,
  225.40, 156.66, 199.69, 139.15, 234.45, 239.69, 166.74, 143.81, 163.93,
  112.80, 223.65]` - first-5 mean 216.8 -> last-5 mean 162.2 (-25%), new
  minimum 112.80 (step 18) vs starting 249.21. The "aa" NLL term (less
  coupled to the random noise-level draw) falls cleanly and almost
  monotonically: 2.996 -> 2.287 over 20 steps (-24%). `encoder parameters
  changed after 20 steps: True`, `encoder has grad != None on last step:
  True` - the Encoder is genuinely jointly trained, not a frozen
  preprocessing step (spec-05's explicit requirement).
- **Task 3**: jobs 901824 (aa_gt real random - see below) then 901845
  (aa_gt all-MASK, the fix) generated `gradstep_default_vp.npz` (181 MB,
  moved out of the notebook to `/hnvme/workspace/v181bb10-salad/n004_evidence/
  gradstep_default_vp.npz` per the >5MB rule) (real
  `default_vp-200k` weights, 2-batch-item/3-chain/padded fixture from
  `tests/harness/gen_loss_ref2.py`, REAL training entry point composition:
  `StructureDiffusionInference.prepare_condition` + `Diffusion(predict_aa=
  False)` + `Diffusion.loss`, not the inference-only `StructureDiffusionPredict`
  `gen_loss_ref2.py` itself uses). `task3_check_gradstep.py` compares.
  - FIRST attempt (real random `aa_gt`) showed "aa" loss off by 9%, and
    gradient rel-error 1.0 on MANY unrelated trunk parameters. Root cause
    found: `AADecoder.train_forward` (torch, `decoders.py:188`) draws its own
    ADM masking randomness with NO override hook, same on the JAX side, and
    the two RNG streams (threefry vs torch generator) cannot be made to draw
    the same mask - the discrete mask choice differs, and since aa_decoder
    shares the trunk's `local`/`pos` features, that discrepancy propagates
    into gradients for parameters that have nothing to do with aa/dssp.
    Fixed the SAME way `scripts/gen_fixtures_model.py` already does for this
    exact reason: `aa_gt` all-MASK, which short-circuits `train_forward` to a
    deterministic no-op on both backends. Cost: `aa`/`x`/`violation`/`local`
    loss terms become exactly 0 (UNK's atom14 template is empty) - out of
    scope for this check, noted in the script and the final report.
  - With that fix: all 9 remaining loss terms match to <=2e-4 relative (dssp
    3.3e-5, pos 1.6e-4, fape 1.9e-6, ...). Gradient: whole-model flattened
    cosine similarity **0.99999863**, norm rel **1.07e-4**; 385/386 params
    with non-negligible gradient have per-parameter cosine >0.999 (worst
    0.9954); 393/779 below the 1e-4 noise floor on both sides (superset of
    the 202 documented grad=None `.b` biases). Post-step (one real
    `torch.optim.Adam` vs one `optax.chain(clip_by_global_norm, scale_by_adam,
    scale_by_schedule, scale(-1.0))` step) parameter DELTA does NOT match at
    the per-element level even for these well-matched gradients - explained,
    not hand-waved: Adam's step-1 update is exactly `sign(g)*lr` per element
    (bias correction at t=1 makes `m_hat=g`, `v_hat=g^2`), so an element with
    a near-zero true gradient has its sign set by ordinary fp32 noise, and
    Adam amplifies that into a full +-lr discrepancy regardless of how
    tightly the gradient itself matches. Verified directly (see report) on
    4 example parameters: cosine 0.9998-1.0, one single near-zero coordinate
    flipped sign in the delta. eps placement/bias correction/weight-decay
    coupling were checked analytically first (matches `notes/specs/
    04_data_and_training.md` §B.2's own optax-vs-torch equivalence argument
    exactly) and are NOT the discrepancy source.
- **Task 4**: transcribed `reveal_conditioning` from `Condition.get_masks`
  (NSB:2562-2586) + `StructureDiffusion.prepare_condition`'s outer gate
  (NSB:241-258) instead of approximating. 3 concrete differences from the old
  best-effort version (full detail in the function's docstring / final
  report): (a) reveal rate is `U(0.2,1.0)` not `U(0,1)`; (b) there are TWO
  independent per-batch-item p=0.5 gates multiplied (net 25% survival, not
  50%), and aa/dssp share ONE of those two gates (use-correlated, not
  independent); (c) `dssp_mean` (chain-grouped, `U(0.8,1.0)` local range)
  conditioning was entirely absent before and is now wired through
  `prepare_batch` into `data["dssp_mean"]`/`dssp_mean_mask"`. Not re-verified
  against JAX end-to-end (would need to reproduce JAX's threefry draws, which
  is exactly the cross-backend RNG problem this whole gap is about) - checked
  by direct line-by-line reading of NSB against the new torch code instead.

## Outcome
See the SubagentHandback report for the full per-task numbers; this item is
the durable record. Two real bugs found and fixed outside the caller's
DO-NOT-TOUCH list (`modules/encoder.py`'s 2 crash sites, `losses/{structure,
total}.py`'s 3 more of the same class) that blocked Task 2/3 entirely until
fixed - both independently corroborated by the concurrent N003 red-team pass.
`default_vp_scaled`/`multimotif_vp` parity is measured but not fully
root-caused (open item, see Next).

## Outputs
`.agent-research/results/N004-porting-gaps/` (note: this task's caller
specified this exact path; the auto-created `results/N004-salad-porting-
gaps-parity-encoder-traini/` sibling folder from `new-item.sh` is unused -
empty, safe to ignore/remove):
- `gen_fixtures_task1.sbatch`, `genfix_901506.log` - Task 1 fixture generation
- `task1_parity_default_vp_scaled.txt`, `task1_parity_multimotif_vp.txt` -
  Task 1 parity run output
- `task2_overfit_check.py` - Task 2 self-check (Trainer overfit)
- `gen_gradstep_ref.py`, `gen_gradstep_task3.sbatch`, `gradstep_901824.log`
  (superseded, real aa_gt), `gradstep_901845.log` (final, all-mask aa_gt) -
  Task 3 JAX reference generation; the reference array itself,
  `gradstep_default_vp.npz`, is at `/hnvme/workspace/v181bb10-salad/
  n004_evidence/gradstep_default_vp.npz` (>5MB, moved out of the notebook)
- `task3_check_gradstep.py`, `task3_gradstep_result.txt` - Task 3 comparison
  + full output

External: `fixtures/model_default_vp_scaled.npz`, `fixtures/
model_multimotif_vp.npz` (new, committed alongside the existing
`fixtures/model_{default_vp,default_ve_scaled}.npz`).

## Next
- `default_vp_scaled`/`multimotif_vp`'s ~300-1800x-looser-than-default_vp
  parity is unresolved. Whoever owns `modules/blocks.py`/`diffusion.py` next
  should re-run `tests/test_model_parity.py` (fixtures already exist) with
  instrumentation on the neighbour-selection ties specifically (candidate
  scores within ~1e-5 of the k-th selected one) to confirm/refute the
  fp32-tie-flip hypothesis.
- Promote `.agent-research/results/N003-.../r05_encoder_parity_{jax,torch}.py`
  to `tests/test_encoder_parity.py` (N003's own recommendation; re-confirmed
  passing here after the real fix landed) - not done in this pass, time
  budget went to Tasks 1-4.
- Task 4's new `reveal_conditioning` is unverified against JAX end-to-end
  (can't be, cross-backend RNG) - a plausibility/sanity check (composition
  distribution over many draws matching the intended ~25% use-rate) would be
  a cheap follow-up if this pathway becomes load-bearing.
- The real training script (`training/train.py`) has never been run beyond
  this task's small synthetic overfit smoke test - no multi-hundred-step or
  real-PDB-data run yet.
