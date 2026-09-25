# SALAD training loss — complete specification

Source of truth: `salad/modules/noise_schedule_benchmark.py` (JAX, read-only,
never edit). All line numbers below refer to that file unless stated
otherwise. Config: `salad/modules/config/noise_schedule_benchmark.py:3`,
variable `default_vp`. This spec is written **for `default_vp` specifically**
(`diffusion_kind="vpfixed"`, `preconditioning="vp"`, `diffusion_time_scale="cosine"`,
`sigma_data=10.0`, `augment_size=15`). Every term states whether it is ACTIVE
or DEAD under this exact config, because the training script contains many
experimental branches that are compiled but never exercised.

Notation used throughout: `L` = total number of residues in the flattened,
multi-item batch (there is no batch dimension — items are concatenated along
axis 0 and distinguished by `batch_index`, shape `(L,)`). `A = 20` is the
number of pseudo-atoms carried through the diffusion trunk for `default_vp`
(`5` backbone/CB atoms + `augment_size=15`). `14` is the fixed atom14 slot
count used for the actual side-chain / all-atom representation — **a
different, unrelated axis from `A`**; do not confuse them. `D = diffusion_depth
= 6` is the number of blocks in the diffusion stack / the trajectory length.
`K = fape_neighbours = 64`, `K' = local_neighbours = 16`.

---

## 0. Two headline findings the writer must not miss

**(a) There is no active SE(3)/Kabsch alignment in `default_vp`.** The
training script contains a full Kabsch-alignment code path (`c.kabsch`,
§4 below) that would align the ground truth onto the prediction before
computing the position losses. In `default_vp`, `kabsch` is never set in the
config, and `dotdict.__getattr__ = dict.get` (see
`salad/modules/utils/collections.py:5`) means `c.kabsch` evaluates to `None`
→ falsy. **All position/atom/FAPE losses below compare raw, un-aligned
world-frame coordinates.** The model itself is responsible for outputting
coordinates in the same global frame as the (already-centered, see §1)
ground truth. Section 4 specifies `index_kabsch`/`index_align` fully anyway,
because other configs (`semi_equivariant_kabsch_vp`, `default_vp_atom14_kabsch`)
turn it on and the writer should implement it once, correctly gated by a
config flag.

**(b) The EDM/flow loss-preconditioning weight is a no-op constant for
`default_vp`.** `preconditioning_scale_factors(c, t_pos, sigma_data)["loss"]`
(NSB:967-987, already ported at `salad_torch/diffusion/precond.py`) is used
exactly once inside `loss()`, to build `diffusion_weight = base_weight *
loss_weight` (NSB:1242-1243). For `c.preconditioning == "vp"` the function
falls into its final `else` branch and returns `{"loss": 1}` — a Python int
`1`, not even a tensor, completely independent of `t`. So for `default_vp`,
`diffusion_weight == base_weight` exactly; **there is no time-dependent
reweighting of the `pos`/`pos_trajectory` losses at all.** This preconditioning
machinery only does something for `preconditioning in ("edm", "edm_scaled")`
(→ `1/out_scale**2`, using `edm_scaling`) or `"flow"` (→ `1/max(t**2, 1e-6)`,
flagged upstream itself as `# FIXME: loss reweighting`, NSB:980). Implement
the general dispatch (it's already ported), but know that for `default_vp`
it contributes nothing beyond returning the literal `1`.

---

## 1. Data flow into the loss

### 1.1 `StructureDiffusion.prepare_data` (NSB:106-160)

Given a raw batch with `all_atom_positions (L,≥14,3)`, `all_atom_mask (L,≥14)`,
`chain_index (L,)`, `batch_index (L,)`, `seq_mask (L,)`, `residue_mask (L,)`:

1. Truncate positions/mask to atom14: `pos = pos[:, :14]`, `atom_mask = atom_mask[:, :14]`.
2. `chain = unique_chain(chain_index, batch_index)` — globally-unique chain
   ids across the whole flattened batch (already ported,
   `salad_torch/geometry/segment.py:117`).
3. `mask = seq_mask * residue_mask * atom_mask[:, :3].all(-1)` — a residue is
   "valid" only if N, CA, C (atom14 indices 0,1,2) are all resolved.
4. **Centering**: `center = index_mean(pos[:, 1], batch, atom_mask[:, 1, None])`
   (mean CA position **per batch item**, masked by whether CA exists);
   `pos = pos - center[:, None]`. This is the *only* centering step in the
   whole pipeline — there is no additional per-step recentering anywhere
   inside the diffusion stack.
5. `pos = where(atom_mask[..., None], pos, compute_pseudo_cb(pos)[:, None, :])`
   — every **missing** atom slot (not just missing CB) is filled with the
   idealised pseudo-CB position of that residue, not zero. `compute_pseudo_cb`
   is already ported (`salad_torch/geometry/frames.py:38`).
6. `equivariance == "semi_equivariant"` data-augmentation branch: **dead for
   `default_vp`** (`equivariance` unset).
7. Store `pos_14 = pos`, `atom_mask_14 = atom_mask` (this becomes
   `data["atom_pos"]` / `data["atom_mask"]`, the ground truth used by the
   all-atom losses, §7.3, §7.6, §7.9). Then call `self.encode(data)`.

### 1.2 `StructureDiffusion.encode` (NSB:162-221) → `data["pos"]` (the *other* ground truth)

This produces the **backbone+augment** representation that the diffusion
trunk actually denoises (shape `(L, A=20, 3)` for `default_vp`), which is a
**separate tensor from atom14** used by a separate set of losses:

1. `backbone = positions_to_ncacocb(pos_14)` → `(L, 5, 3)`, order N, CA, C, O,
   idealised-CB (already ported, `salad_torch/geometry/frames.py:49`).
2. `equivariance == "semi_equivariant"`: dead branch, skip.
3. Since `c.augment_size = 15 > 0`: run the `Encoder` module (NSB:694-773,
   a **model** component, out of scope for this loss spec) which appends 15
   extra pseudo-atom positions, frame-anchored, learned via `VectorLinear` +
   an `EncoderStack` refinement — producing `pos = augmented_pos` of shape
   `(L, 20, 3)`.
4. `c.encode_atom14` / `c.atom14_ca` / `c.atom14_learned`: dead (`encode_atom14`
   unset).
5. Return `seq = aa_gt` (ground truth amino acid indices, `(L,)` int, values
   0-19 for the 20 canonical residues, 20 = "unknown/gap" used as the mask
   token elsewhere), `pos` (the `(L, 20, 3)` tensor, **this becomes
   `data["pos"]`, the ground truth consumed by every backbone/augment-space
   loss**: `pos`, `pos_trajectory`, `rotation`, `rotation_trajectory`, `fape`,
   `fape_trajectory`, `distogram`, and it also supplies the CB used to build
   `local_neighbours`).

So: **`data["pos"]` (20 atoms, encoder output) and `data["atom_pos"]` (14
atoms, raw atom14 ground truth) are two different ground-truth tensors**,
consumed by disjoint sets of loss terms. Do not merge them.

### 1.3 `prepare_condition` (NSB:223-342)

Produces `data["condition"]`, `data["dssp_gt"]`, `data["aa_mask"]`,
`data["dssp_mask"]`, `data["pair_condition"]`. Only the following fields feed
the loss (the rest is model conditioning input, out of scope here):
- `dssp_gt (L,)` int ∈ {0,1,2} (3-state secondary structure), ground truth
  for the `dssp` loss (§7.2).
- `aa_mask (L,)` bool — True where the amino-acid identity was **given** to
  the model as a hard condition (stochastic, `p=0.5` overall use, then a
  further per-residue Bernoulli inside `Condition`). Positions with
  `aa_mask=True` are **excluded** from the `aa` NLL loss (§7.1) — the model
  isn't asked to predict what it was already told.
- `dssp_mask (L,)` bool — same idea for secondary-structure conditioning;
  excluded from the `dssp` loss.

### 1.4 `apply_diffusion` (NSB:344-410, `default_vp` branch only)

This is the **forward noising process**, called once per training example
(and again, with fresh randomness but the *same* `t_pos`/`t_seq` if already
present in `data`, inside the self-conditioning loop — see §1.5). For
`diffusion_kind in ("vp","vpfixed","flow","flow_scaled")`:

```text
t_pos = uniform(0,1) per BATCH ITEM, broadcast to residues via [batch_index]   # (L,)
sigma  = SIGMA_SCALE["cosine"](t_pos)                                          # sigma_scale_cosine, below
t_pos  = sigma                        # <-- t_pos is REASSIGNED to the schedule output.
                                       #     From here on "t_pos" means "noise scale sigma
                                       #     in the cosine-VP sense", NOT raw uniform time.
t_seq  = uniform(0,1) per batch item, independent draw, broadcast via [batch_index]
if "t_pos" already in data:           # true only on self-conditioning re-entry, see §1.5
    t_pos, t_seq = data["t_pos"], data["t_seq"]   # reuse the SAME noise level,
                                                   # a FRESH noise instance is still drawn below
cloud_std = c.sigma_data = 10.0       # because diffusion_kind == "vpfixed"
pos_noised = diffuse_atom_cloud(key, pos, mask, batch, t_pos[:, None],
                                 cloud_std=10.0, flow=False)     # flow=False: diffusion_kind != "flow"/"flow_scaled"
seq_noised, corrupt_aa = diffuse_sequence(key, seq, t_seq)
```

`sigma_scale_cosine(t, s=0.01)` (NSB:62-75):
```text
alpha_bar = cos((t+s)/(1+s) * pi/2)^2 / cos(s/(1+s) * pi/2)^2      # normalised so alpha_bar(0)=1
alpha_bar = clip(alpha_bar, 0, 1)
sigma     = sqrt(1 - alpha_bar)
```

`diffuse_atom_cloud` (`salad/modules/utils/diffusion.py:69-114`, **not yet
ported**, port it exactly):
```text
target = cloud_std * randn(L, A, 3)                       # cloud_std broadcasts, here a scalar 10.0
center = per-batch-item mean of target[:, 1] (CA slot), scattered back to every row via batch_index
target = target - center[:, None, :]                      # re-center the NOISE cloud itself, not x
result = x * sqrt(1 - t**2) + target * t                   # flow=False branch (VP interpolation)
```
Note the noise cloud is generated **fresh every call** (new `randn`) and is
independently re-centered per batch item — this re-centering of the *noise*,
not of `x`, is easy to invert accidentally; keep it as a separate step.

`diffuse_sequence` (`salad/modules/utils/diffusion.py:12-25`, not yet ported):
```text
corrupt_aa = bernoulli(p=t, shape=x.shape)     # per-residue independent draw, prob = t_seq of THAT residue's batch item
out = where(corrupt_aa, 20, x)                 # 20 = mask token
```

`get_sigma_edm` (NSB:663-691) — **DEAD for `default_vp`** (only used when
`diffusion_kind == "edm"`). Specify for completeness since it was named as a
primary source: samples `sigma = exp(meanval + stdval * randn)` per batch
item if `meanval` is given (log-normal), else `sigma = minval + uniform*(maxval-minval)`.
Not exercised by `default_vp`; do not wire it up unless a VE config is added.

### 1.5 Self-conditioning (NSB:412-439) — context, not a loss term

`StructureDiffusion.__call__` runs the diffusion stack 0 or 1 times first
(`count = randint(0,2)`, i.e. a coin flip) under `stop_gradient`, to produce
`prev = {"pos", "local"}` self-conditioning features, **then** runs the real
forward pass that produces `result`, on which `loss()` is called. This
affects what the model *sees* as input but not the loss formulas below
(`data["pos"]`, `data["atom_pos"]`, `data["t_pos"]` etc. used in `loss()` are
always the *original* outer-scope values, not the self-conditioning preview's).

---

## 2. `late_mask` (NSB:1190-1197)

```text
late_mask = (t_pos < 5.0)   if diffusion_kind == "edm"      # dead for default_vp
            (t_pos < 0.5)   otherwise                        # default_vp branch
late_mask *= mask
```
Recall `t_pos` here is the **cosine-schedule sigma output** (§1.4), not raw
uniform time — so "late" means "low noise level" (`sigma < 0.5`), i.e. the
tail of the reverse (denoising) trajectory close to the clean structure, not
literally the last training-time-index. Used by: `local` (§7.7), `violation`
(§7.9), and (dead) `x_late`.

## 3. `base_weight` / `get_batch_weight` — the canonical per-item average (NSB:1232-1235, `salad/modules/utils/losses.py:42-55`)

```text
base_weight = mask / max(index_sum(mask.float(), batch, mask), 1) / (batch.max() + 1)
```
`index_sum(mask, batch, mask)` is, per residue, the **count of valid residues
in that residue's batch item** (broadcast back to every member — already
ported, `salad_torch/geometry/segment.py:26`). So `base_weight[i] = 1 /
(item_size(i) * num_items)` for valid `i`, `0` for invalid `i`. Summing
`base_weight` over ALL `L` residues in a loss term therefore gives an
**unweighted mean over batch items** of each item's own per-residue mean —
every protein in the batch contributes equally to the loss regardless of its
length. **This is the reduction used by `pos`, `pos_trajectory`, `x`,
`rotation`, `rotation_trajectory`, `fape`, `fape_trajectory`, `local`, and
`distogram`.**

Contrast with `aa` and `dssp` (§7.1, §7.2), which instead do
`sum(masked_nll) / max(sum(predict_mask), 1)` — a **flat global average over
every valid residue in the whole flattened batch, with no per-item
equalisation.** Longer proteins in a batch dominate the `aa`/`dssp` losses
proportionally to their residue count; they do not dominate any of the
`base_weight`-reduced losses. This asymmetry is real and present in the
reference; reproduce it exactly, do not "fix" it by making `aa`/`dssp`
consistent with the rest.

`violation` (§7.9) uses yet a third pattern: its own internal
sums-over-everything, no `base_weight`, no batch-size normalisation at all
beyond the internal `/(count+eps)` terms baked into
`custom_structural_violation_loss`/`custom_clash_loss`/`between_residue_bond_loss`
(§7.9 details this).

## 4. `index_kabsch` / `index_align` — full spec (DEAD for `default_vp`, gated by `c.kabsch`)

Not yet ported. `salad/modules/utils/geometry.py:647-701`. Signature:
`index_align(x, y, index, mask, weight=None) -> x_aligned` — **rigidly
transforms `x` onto `y`**, i.e. **the first argument is moved, the second is
the fixed target**. Where used in `loss()` (NSB:1247-1248, NSB:1277-1278):
```text
if c.kabsch:
    pos_gt      = index_align(pos_gt,      pos_pred,      batch, mask)
    atom_pos_gt = index_align(atom_pos_gt, atom_pos_pred, batch, mask)
```
**The ground truth is aligned onto the prediction, not the other way
around.** This is the opposite of the usual convention (align prediction to
target) — if/when this flag is ever turned on, the aligned quantity fed into
the downstream MSE is a *moving* target that chases the prediction's current
global pose.

`index_kabsch(x, y, index, mask, weight=None) -> (rot, x_center, y_center)`:
```text
x_center = index_mean(x, index, mask[:, None], weight)      # per-item mean, over ALL atoms passed in (see below: only CA is passed)
y_center = index_mean(y, index, mask[:, None], weight)
x -= x_center;  y -= y_center
w = weight if weight is not None else ones_like(index)[:, None]
covariance = index_sum(w[:, :, None] * x[:, :, None] * y[:, None, :], index, mask[:, None, None])   # (num_items, 3, 3), outer product x⊗y summed per item
u, _, v = svd(stop_gradient(covariance), full_matrices=True)     # <-- SVD input is DETACHED
det_sign  = det(u) * det(v)
det_flip  = ones(num_items, 3); det_flip[:, -1] = det_sign        # reflection-correction on the last singular vector, standard Kabsch/Umeyama fix
rot = einsum("...ak,...kb->...ba", u * det_flip[:, None, :], v)    # proper rotation, det=+1
return rot, x_center, y_center
```
`index_align(x, y, index, mask, weight=None)`:
```text
rot, x_center, y_center = index_kabsch(x[:, 1], y[:, 1], index, mask, weight)   # <-- ONLY the CA slot (index 1) is used to determine the alignment
result = einsum("...ak,...ik->...ia", rot, x - x_center[:, None]) + y_center[:, None]   # rotation applied to ALL atoms of x (not just CA)
```
**Gradient flow, precisely**: `jax.lax.stop_gradient` wraps only the
`covariance` tensor fed to `svd`. Consequences:
- `rot` carries **zero gradient** to either `x` or `y` — the optimal rotation
  is treated as a constant w.r.t. both inputs, even though it was computed
  *from* both of them.
- `x_center` (from `pos_gt`) and `y_center` (from `pos_pred`) are **not**
  detached — gradients flow normally through both.
- The final `result` therefore receives gradient from `pos_pred` **only**
  through the additive `y_center` term (a per-item constant translation, no
  rotational coupling), and from `pos_gt` through both `x_center` and the
  `(x - x_center)` term rotated by the (frozen) `rot`.
- Net effect if ever enabled: the loss can move the *prediction's* alignment
  target only by translating it (via `y_center`); it cannot get gradient
  signal for improving rotational agreement through this path at all — any
  rotational learning signal has to come from elsewhere (e.g. the separate
  `rotation` loss, §7.4).

In PyTorch: `torch.linalg.svd(covariance.detach(), full_matrices=True)`
(note: JAX's `svd(..., compute_uv=True)` returns `u, s, vh`, i.e. `v` in the
JAX code above is **already `vh`** — `torch.linalg.svd` returns the same
`(U, S, Vh)` convention, so this ports directly with no transpose fix-up).

## 5. `p_clip` / `do_clip` (NSB:1237-1240)

```text
do_clip = where(bernoulli(p=c.p_clip, shape=batch.shape)[batch], 100.0, inf)
```
`c.p_clip = 1.0` for `default_vp` ⇒ `bernoulli(p=1.0, ...)` is **always
True** ⇒ `do_clip` is the constant `100.0` for every residue, every step —
the stochastic gate never actually fires for `default_vp` (it exists to be
tuned down in other configs to randomly *disable* clipping some fraction of
the time). Used as a squared-distance clip (`100.0 Å² = (10 Å)²`) in `pos`
and `pos_trajectory`, and as a squared-distance clip in `x` as well (same
`100.0`, see §7.3 — the *local*-FAPE clip at §7.7 is a **different**,
unsquared `10.0` bound, see the note there).

---

## 6. Loss weight table (`default_vp`)

| loss key | config field | value | hardcoded? | active for `default_vp`? |
|---|---|---|---|---|
| `aa` | `aa_weight` | 10.0 | no | **yes** |
| `aa_trajectory` | `aa_weight` (shared) | 10.0 | no | **DEAD** (`c.aa_trajectory` unset) |
| `dssp` | — | 1.0 | **yes, literal `1.0`** | **yes** |
| `pos` | `pos_weight` | 2.0 | no | **yes** |
| `pos_trajectory` | `trajectory_weight` | 1.0 | no | **yes** |
| `x` | `x_weight` | 2.0 | no | **yes** |
| `rotation` | `rotation_weight` | 2.0 | no | **yes** |
| `rotation_trajectory` | `rotation_trajectory_weight` | 1.0 | no | **yes** |
| `fape` (final step) | `fape_weight` | 1.0 | no | **yes** |
| `fape_trajectory` | `fape_trajectory_weight` | 0.5 | no | **yes** |
| `local` | `local_weight` | 10.0 | no | **yes** |
| `distogram` | — | 0.1 | **yes, literal `0.1`** | **yes** |
| `lddt` (soft-LDDT) | `local_weight` (shared) | 10.0 | no | **DEAD** (`c.soft_lddt` unset) |
| `distogram_trajectory` | — | 0.1 | **yes, literal `0.1`** | **DEAD** (`c.distogram_trajectory` unset) |
| `violation` | `violation_weight` | 0.1 | no | **yes** |

Total objective for `default_vp`, in the exact order the reference
accumulates it (NSB:1206, 1229, 1262, 1272, 1289, 1297, 1307, 1335, 1352,
1367, 1431):

```text
total = 10.0 * aa
      +  1.0 * dssp
      +  2.0 * pos
      +  1.0 * pos_trajectory
      +  2.0 * x
      +  2.0 * rotation
      +  1.0 * rotation_trajectory
      + (1.0 * fape + 0.5 * fape_trajectory) / 3      # NB: this /3 wraps BOTH terms together, see §7.5
      + 10.0 * local
      +  0.1 * distogram
      +  0.1 * violation
```

---

## 7. Term-by-term specification

### 7.1 `aa` — amino-acid NLL (NSB:1199-1206) — **ACTIVE**

Config: `aa_weight = 10.0`.

```text
aa_predict_mask = mask * (aa_gt != 20) * result["corrupt_aa"]        # (L,) bool
aa_predict_mask = where(aa_mask, 0, aa_predict_mask)                  # exclude conditioned residues
aa_nll = -sum_c( result["aa"][:, c] * one_hot(aa_gt, 20)[:, c] )      # (L,), result["aa"] is LOG-probabilities (20 classes)
aa_nll = where(aa_predict_mask, aa_nll, 0)
aa = sum(aa_nll) / max(sum(aa_predict_mask), 1)                       # GLOBAL average, no base_weight (see §3)
```

- **Computed against**: `result["aa"]` — log-softmax output of `AADecoder`
  (NSB:1497-1520, a model head, out of scope), **20** classes (cannot predict
  the mask token itself); ground truth `data["aa_gt"] (L,)` int ∈ [0,20)
  (20 = unresolved/unknown, excluded via `aa_gt != 20`).
- **`result["corrupt_aa"]` is NOT `data["t_seq"]`-driven.** For
  `aa_decoder_kind == "adm"` (default_vp), `AADecoder.train` (NSB:1548-1563)
  draws its **own, independent** per-batch-item uniform time `t = uniform(0,1)[batch]`
  and calls `diffuse_sequence(key, aa_gt, t)` internally — a *third*,
  separate random masking process, distinct from both `data["t_seq"]` (used
  only for the model-input sequence features, §1.4, which for `default_vp`
  are never even fed to the trunk — see next point) and from any structural
  `t_pos`. `result["corrupt_aa"]` (as consumed by the loss) is then further
  masked: `result["corrupt_aa"] = corrupt_aa_from_decoder * data["mask"] *
  (data["aa_gt"] != 20)` (NSB:1085).
- **`default_vp` never diffuses sequence into the trunk.** `c.diffuse_sequence
  = False` for `default_vp`, so the `if c.diffuse_sequence:` block in
  `Diffusion.prepare_features` (NSB:1163-1165) is skipped entirely — the
  diffusion trunk receives **no sequence information whatsoever** (noised or
  clean). The entire amino-acid prediction task is delegated to the
  self-contained ADM-style `AADecoder`, which reads only the trunk's output
  `local`/`pos` features plus its own independently-masked copy of `aa_gt`.
  Do not wire `data["seq_noised"]`/`data["t_seq"]` into the trunk features
  for `default_vp`.
- **Mask**: `mask` (residue validity) ∧ `aa_gt != 20` (resolved identity) ∧
  `corrupt_aa` (this residue was one of the ADM-masked positions) ∧ NOT
  `aa_mask` (not given as a hard condition).
- **Reduction**: flat global average (§3), **not** `base_weight`.
- **No clamp, no explicit epsilon** beyond the `max(..., 1)` denominator
  guard.

### 7.2 `aa_trajectory` — DEAD (NSB:1209-1221)

Only computed `if "aa_trajectory" in result`, which requires `c.aa_trajectory`
truthy (selects a different code path in `Diffusion.__call__`, NSB:1074-1076)
— unset for `default_vp`. Specified for completeness: same NLL as §7.1 but
over every trajectory step, weighted by a **geometric decay towards the
earliest step** `0.99 ** arange(D)[::-1]` (so the *last* step gets weight
`0.99^0 = 1`, the first gets `0.99^(D-1)`), then `.mean()` over steps. This
is the **only** trajectory loss in the whole file that uses a decayed
average — `pos_trajectory`, `rotation_trajectory`, `fape_trajectory`,
`distogram_trajectory` all use a **plain, unweighted `.mean()`** over the `D`
steps. Do not implement unless `aa_trajectory` is later turned on.

### 7.3 `dssp` — secondary-structure NLL (NSB:1223-1229) — **ACTIVE**

Weight: hardcoded `1.0` (not a config field — grep for `total += 1.0 *
dssp_nll`, there is no `dssp_weight` key anywhere in `default_vp`).

```text
dssp_predict_mask = mask * (1 - dssp_mask)                          # exclude conditioned residues
dssp_nll = -sum_c( result["dssp"][:, c] * one_hot(dssp_gt, 3)[:, c] )  # result["dssp"] = log_softmax(Linear(3, bias=False)(local)), a BARE linear probe on the trunk's final `local` features — no MLP, no decoder stack
dssp_nll = where(dssp_predict_mask, dssp_nll, 0)
dssp = sum(dssp_nll) / max(sum(dssp_predict_mask), 1)                # GLOBAL average, same pattern as §7.1
```
Ground truth `dssp_gt` comes from `assign_dssp` (`salad/modules/utils/dssp.py`,
out of scope — it's a labelling function that runs once in
`prepare_condition`, not part of the loss).

### 7.4 `pos` / `pos_trajectory` — backbone+augment diffusion MSE (NSB:1231-1272) — **ACTIVE**

Config: `pos_weight = 2.0`, `trajectory_weight = 1.0`.

```text
diffusion_weight = base_weight * preconditioning_scale_factors(c, data["t_pos"], sigma_data)["loss"]
                  = base_weight * 1.0                                    # constant for preconditioning=="vp", see §0(b)

pos_gt   = data["pos"]                              # (L, 20, 3) — encoder-produced ground truth, §1.2
pos_pred = result["pos"]                            # (L, 20, 3) — final trunk output
if c.kabsch: pos_gt = index_align(pos_gt, pos_pred, batch, mask)   # DEAD, §4

pos_mask = mask[:, None] * ones_like(pos_gt[..., 0], bool)     # (L, 20) — broadcasts `mask` over ALL 20 atom slots equally;
                                                                 #           there is no per-atom validity mask here (all 20
                                                                 #           encoder-space atoms are always "present" once the
                                                                 #           residue itself is valid — the encoder invented them)
if c.mask_atom14: pos_mask *= data["atom_mask"]                 # DEAD (mask_atom14 unset)

dist2 = sum_xyz( (pos_pred - pos_gt) ** 2 )                     # (L, 20)  — squared Euclidean distance per atom
dist2 = dist2 * pos_mask
dist2 = clip(dist2, 0, do_clip) . sum(atom_axis) / max(pos_mask.sum(atom_axis), 1)   # (L,) — clip PER ATOM to 100.0 Å² (§5), THEN sum over the 20 atoms, THEN divide by valid-atom count -> mean clipped squared-distance per atom for this residue
dist2 = where(mask, dist2, 0)
pos = sum( dist2 * diffusion_weight ) / 3                       # base_weight-style item-equalised sum, THEN a global /3
```
The final `/ 3` converts "sum of 3 squared coordinate errors" into an
effective per-coordinate MSE (`E[dx²+dy²+dz²] / 3 ≈ E[dx²]`), applied once
at the very end after the weighted sum, not per-residue — mathematically
equivalent either way since it's a global linear scale, but keep the
division exactly where the reference puts it if you want bit-identical
intermediate `losses[...]` values for debugging against JAX.

`pos_trajectory` (NSB:1263-1272): identical formula, applied to
`result["trajectory"] (D, L, 20, 3)` broadcast against `pos_gt[None]`,
`do_clip[None, :, None]`, `mask[None]`; the residue-axis reduction produces
`(D, L)`, weighted-summed over `L` (`diffusion_weight[None, ...]` broadcasts
`(L,)` against `(D, L)`) to get `(D,)`, `/3` applied per-step **inside** the
per-step formula (`... / 3` appears before `.mean()`), then `.mean()` over
the `D` axis — a **plain, unweighted** average over trajectory steps (see
§7.2 for the one exception that isn't plain).

`jax.vmap` note: nowhere in this specific block is `vmap` used — the
trajectory tensor's leading `D` axis is handled by ordinary broadcasting of
elementwise ops (`**2`, `sum`, `clip`, `*`). No translation needed beyond
"the extra leading axis just works" in torch too.

### 7.5 `x` — all-atom (atom14) diffusion MSE (NSB:1273-1289) — **ACTIVE**

Config: `x_weight = 2.0`. Ground truth `data["atom_pos"] (L, 14, 3)` (§1.1,
the RAW atom14 tensor, not the encoder output). Prediction
`result["atom_pos"] (L, 14, 3)`, produced by `get_angle_positions` (NSB:1435-
1457, a model head — **important context, not itself a loss term**): atoms
0-3 (N, CA, C, O) are copied verbatim from `result["pos"][..., :4, :]` (the
same predicted backbone as §7.4); atoms 4-13 (idealised-CB + side chain) are
built by applying **predicted** torsion angles (a small MLP regressing 7
`(sin, cos)` pairs, each re-normalised to the unit circle) to an idealised
rotamer template selected by the **ground-truth** amino acid identity
`data["aa_gt"]` (via the already-ported `single_protein_sidechains`,
`salad_torch/geometry/sidechain.py:429`) — **never** the predicted/sampled
aatype, even at inference (`predict_aa=True` still passes `aatype` only into
`single_protein_sidechains`, but the atom14 ground-truth identity used here
during training is always `aa_gt`). Consequence: none of `x`, `rotation`,
`local`, or `violation` (§7.5-7.9) ever penalise the model for *sequence*
uncertainty in atom placement beyond the 7-angle regression — the rotamer
*topology* is always correct-by-construction during training.

```text
if c.kabsch: atom_pos_gt = index_align(atom_pos_gt, atom_pos_pred, batch, mask)  # DEAD, §4
dist2 = sum_xyz( (atom_pos_gt - result["atom_pos"]) ** 2 )        # (L, 14)
dist2 = clip(dist2, 0, do_clip)                                   # per-atom clip to 100.0 Å² — SAME bound and SAME do_clip draw as §7.4 (not resampled)
atom_mask = data["atom_mask"]                                     # (L, 14) — EMPIRICAL atom14 mask (was this atom resolved in the crystal structure), not the canonical per-restype mask
if c.x_late: atom_mask *= late_mask[..., None]                    # DEAD (x_late unset)
x = sum( where(atom_mask, dist2, 0), atom_axis ) / 3               # sum over 14 atoms, /3 for per-coordinate scaling
x = x / max(atom_mask.sum(atom_axis), 1)                           # THEN divide by valid-atom count — order differs from §7.4 (there /count happens before /3) but is mathematically identical since both are just scalar divisions of the same numerator
x = sum( x * base_weight )                                         # item-equalised reduction (§3) — NOTE: uses base_weight here, unlike aa/dssp
```

### 7.6 `rotation` / `rotation_trajectory` (NSB:1290-1307, `salad/modules/utils/losses.py:57-88`) — **ACTIVE**

Config: `rotation_weight = 2.0`, `rotation_trajectory_weight = 1.0`.
Frames are extracted from **atom14** positions here (not the 20-atom
backbone+augment tensor) — `extract_aa_frames` uses only slots 0,1,2 (N, CA,
C) regardless of which tensor it's given, so this is equivalent in practice,
but be precise about which tensor each block starts from:

```text
gt_frames, _ = extract_aa_frames(data["atom_pos"])                 # ground truth atom14 -> Rigid, shape (L,)
frames,    _ = extract_aa_frames(result["atom_pos"])                # predicted atom14   -> Rigid, shape (L,)
rel = gt_frames.rotation.inverse() @ frames.rotation                # R_gt^T @ R_pred, (L, 3, 3)  ("inverse" of a pure rotation = transpose)
rot_loss = sum_{i,j}( (rel - I_3)[i,j] ** 2 )                        # Frobenius norm² of deviation from identity, (L,)
rot_loss = where(mask, rot_loss, 0)
rotation = sum( rot_loss * base_weight )                            # item-equalised, §3. No /3 (rotation matrix entries are unitless).
```
`rotation_trajectory` (NSB:1299-1307): `frames` recomputed from
`result["trajectory"] (D, L, 20, 3)` (note: this trajectory tensor, not
atom_pos — but again only slots 0,1,2 matter and those are shared with
`atom_pos`'s first 4 slots by construction, §7.5); **`gt_frames` is reused
unchanged from the block above** (`gt_frames.rotation.inverse()[None]`,
broadcast — the ground-truth frame does not vary over the trajectory). The
`jax.vmap(extract_aa_frames)` call here has **no torch-side translation
work**: `extract_aa_frames` (already ported,
`salad_torch/geometry/frames.py:24`) is written with `...`-generic
broadcasting throughout (`Rigid.from_reference` → `rot_from_two_vectors` →
elementwise `vec_cross`/`vec_dot`/`vec_normalized` all operate on the last
axis only), so calling it directly on a `(D, L, 20, 3)` tensor produces a
`(D, L)`-shaped `Rigid` with no vmap/loop needed. `Rigid.__getitem__`
likewise generalises to any number of leading axes via its `key + (slice
+ slice(None)*n)` construction (already used later at §7.7 with 2 extra
axes). Reduction: sum over `L` with `base_weight[None]` → `(D,)`, then plain
`.mean()` over `D` (no decay, contrast §7.2).

### 7.7 `fape` / `fape_trajectory` — sparse neighbour FAPE, **squared** (NSB:1309-1335) — **ACTIVE**

Config: `fape_weight = 1.0`, `fape_trajectory_weight = 0.5`, `fape_neighbours
= 64`. Operates entirely in the **20-atom backbone+augment space**
(`data["pos"]` / `result["trajectory"]`), not atom14.

```text
pair_mask = (batch[:, None] == batch[None, :]) * mask[:, None] * mask[None, :]     # (L, L)
pos_gt = where(mask[:, None, None], data["pos"], 0)          # zero out INVALID residues' entire 20-atom row (avoids inf propagating from later distance calc)
frames_gt, _ = extract_aa_frames( stop_gradient(pos_gt) )     # NOTE: explicit stop_gradient on the ground truth here (defensive/no-op in practice since GT carries no grad, but replicate the .detach() for fidelity)
distance = ||pos_gt[:, None, 4] - pos_gt[None, :, 4]||          # (L, L) CB-CB distance (slot 4 = idealised CB in the 5+augment ordering)
distance = where(pair_mask, distance, inf)
neighbours = get_random_neighbours(64)(distance, batch, mask)   # (L, 64) — a FRESH Gumbel-top-k draw, INDEPENDENT of result["distogram_neighbours"] (see §7.8 — same count, different RNG call, different purpose)
mask_neighbours = (neighbours != -1) * mask[:, None] * mask[neighbours]     # (L, 64)

pos_gt_local = frames_gt[:, None, None].apply_inverse_to_point(pos_gt[neighbours])   # (L, 64, 20, 3): neighbour j's full 20-atom set, expressed in residue i's OWN ground-truth frame
traj = result["trajectory"]                                    # (D, L, 20, 3)
frames, _ = extract_aa_frames(traj)                              # (D, L) — direct call, no vmap needed (see §7.6)
traj_local = frames[:, :, None, None].apply_inverse_to_point(traj[:, neighbours])    # (D, L, 64, 20, 3) — same `neighbours` index reused at every trajectory step

sq_err = clip( ||traj_local - pos_gt_local[None]||² , 0.0, 100.0 )    # SQUARED distance (`.norm2()`), clipped to 100.0 Å² == (10 Å)². (D, L, 64, 20) -> mean over the 20-atom axis -> (D, L, 64)
sq_err = where(mask_neighbours[None], sq_err, 0)
per_residue = sum(sq_err, neighbour_axis) / max(sum(mask_neighbours, axis=1)[None], 1)   # (D, L)
per_step = sum(per_residue * base_weight)                        # base_weight (L,) broadcasts against (D, L) -> (D,)

fape            = per_step[-1] / 3            # ONLY the final trajectory step
fape_trajectory = mean(per_step) / 3          # plain mean over all D steps (no decay)
fape_loss_total = (fape_weight * per_step[-1] + fape_trajectory_weight * mean(per_step)) / 3   # NOTE: the /3 is applied ONCE to the weighted SUM in the reference (NSB:1334), not separately to each term — same numeric result as dividing each losses[...] entry by 3 first since it's linear, but if you log `losses["fape"]`/`losses["fape_trajectory"]` for parity checks, apply /3 to each as shown, and ALSO apply it to the combined term fed into `total`.
```
This **is** squared-distance FAPE (`.norm2()`, no square root), clipped at
`100.0` — contrast explicitly with §7.8 below.

### 7.8 `local` — all-atom local-neighbourhood FAPE, **unsquared** (NSB:1337-1352) — **ACTIVE**

Config: `local_weight = 10.0`, `local_neighbours = 16`. Reuses the exact same
`distance` (20-atom-space CB-CB matrix) and `pair_mask` computed in §7.7 —
do not recompute.

```text
local_neighbours = get_neighbours(16)(distance, pair_mask)                # (L, 16) — DETERMINISTIC k-NN (argsort), not random, and NOT gated on residue validity beyond pair_mask
gt_pos_neighbours  = data["atom_pos"][local_neighbours]                    # (L, 16, 14, 3)  <- ATOM14 ground truth, gathered by the neighbour indices found from the 20-atom-space distance matrix
pos_neighbours     = result["atom_pos"][local_neighbours]                  # (L, 16, 14, 3)
mask_neighbours = (local_neighbours != -1)[..., None] * data["atom_mask"][local_neighbours]   # (L, 16, 14)
gt_local = gt_frames[:, None, None].apply_inverse_to_point(gt_pos_neighbours)    # gt_frames = atom14 ground-truth frames from §7.6 (reused, NOT recomputed)
local_positions = frames[-1, :, None, None].apply_inverse_to_point(pos_neighbours)  # frames[-1] = the LAST trajectory step's predicted frames from §7.7 (== frames of result["pos"], since trajectory[-1] == result["pos"])

dist = ||local_positions - gt_local||                    # UNSQUARED Euclidean distance. Despite being named `dist2` in the
                                                            # reference (NSB:1346), THIS IS NOT SQUARED -- it is `.norm()`, not
                                                            # `.norm2()`. Flag this misleading variable name explicitly; do not
                                                            # square it when porting.
dist2 = clip(dist, 0, 10.0)                                # clip the raw distance (Å) at 10.0, NOT at 100.0 -- same physical
                                                            # bound as §7.7's sqrt(100)=10, but applied to the unsquared value here.
local = sum( where(mask_neighbours, dist2, 0), axis=(neighbour,atom) ) / 3       # sum over BOTH the 16-neighbour AND 14-atom axes jointly, /3
local = local / max(mask_neighbours.float().sum(axis=(neighbour,atom)), 1)       # divide by count of valid (neighbour,atom) pairs
local = sum( local * base_weight * late_mask )                                    # item-equalised AND gated by late_mask (§2) -- NOT renormalised
                                                                                     # for the reduced item count when late_mask drops residues; if a whole
                                                                                     # batch item happens to draw t_pos >= 0.5 (early/noisy), its `local`
                                                                                     # contribution is simply zero for that step, not redistributed.
```
This loss is an **L1-in-distance (not L2) FAPE**, and it is **the only
diffusion-family structural loss gated by `late_mask`** among the ones
active for `default_vp` (contrast `pos`/`x`/`rotation`/`fape`, which apply
at every noise level).

### 7.9 `distogram` / `distogram_trajectory` (NSB:1354-1413, `salad/modules/utils/losses.py:90-124` for the generic form) — `distogram` **ACTIVE**, `distogram_trajectory` **DEAD**

Config: weight hardcoded `0.1` for both (no config key). `result["distogram"]`
is produced by `DistogramDecoder` (NSB:1460-1494, a model head; log-softmax
over 64 bins spanning `[0, 22]` Å) using `result["distogram_neighbours"]`, a
`(L, 64)` neighbour index computed **once**, earlier in `Diffusion.__call__`
(NSB:1043-1045): `get_random_neighbours(64)(pos_gt[:, 4], batch, mask)` where
`pos_gt = Vec3Array.from_array(data["pos"])` — the **encoder** CB. **This is
a separate random draw from the `neighbours` computed inside `loss()` for
§7.7's sparse FAPE** — same count (64), same selection function, but two
independent Gumbel-top-k draws with two different RNG keys, computed at two
different points in the forward pass. Do not conflate them or reuse one for
the other.

```text
neighbours = result["distogram_neighbours"]                      # (L, 64), fixed for this forward pass
pair_mask = (neighbours != -1) * mask[:, None] * mask[neighbours]
cb = pos_gt[:, 4]                                                  # pos_gt here is the SAME local variable from §7.7 (data["pos"], zeroed at invalid residues) -- still in scope
distance_gt = ||cb[:, None] - cb[neighbours]||
distogram_gt = stop_gradient( distance_one_hot(distance_gt) )      # 64-bin one-hot, range [0,22], already ported (salad_torch/geometry/features.py:80)
distogram_nll = -sum_bins( result["distogram"] * distogram_gt )    # (L, 64)
distogram_nll = sum( where(pair_mask, distogram_nll, 0), axis=neighbour ) / max(sum(pair_mask, axis=neighbour), 1)   # (L,)
distogram = sum( distogram_nll * base_weight )                      # item-equalised (§3) -- unlike aa/dssp, THIS one does use base_weight
```
`distogram_trajectory` (NSB:1404-1413) — DEAD, `c.distogram_trajectory`
unset; same formula applied per trajectory step with 16 bins (not 64:
`distance_one_hot(distance_gt, bins=16)`), plain `.mean()` over `D`. Only
implement if that flag is added later.

`soft_lddt` / `lddt` (NSB:1369-1402) — DEAD (`c.soft_lddt` unset). Shares
`local_weight` as its coefficient (`total += c.local_weight * (1 -
mean_lddt)`) — note this is `local_weight`, the same key as §7.8, not a
separate `lddt_weight`. Not specified further; not exercised by `default_vp`.

### 7.10 `violation` — structural violation loss (NSB:1415-1431) — **ACTIVE**

Config: `violation_weight = 0.1`. Calls `violation_loss` from
`salad/modules/utils/alphafold_loss.py:337-376` with `clash_overlap_tolerance
=1.5`, `violation_tolerance_factor=2.0`, `per_residue=False`.

```text
res_mask  = mask * late_mask                                       # BOTH validity AND late-only (§2) -- like §7.8, this loss is skipped at high noise
pred_mask = get_atom14_mask(aa_gt) * res_mask[:, None]              # (L, 14) -- CANONICAL per-restype atom mask (does this amino acid HAVE this atom), NOT data["atom_mask"] (whether it was resolved in the crystal). These usually agree but are conceptually different masks; use the canonical one here to match upstream exactly.
violation, _ = violation_loss(aa_gt, residue_index, result["atom_pos"], pred_mask, res_mask,
                               clash_overlap_tolerance=1.5, violation_tolerance_factor=2.0,
                               chain_index=data["chain_index"], batch_index=data["batch_index"], per_residue=False)
total += 0.1 * violation.mean()   # violation_weight, and .mean() is a no-op: violation is already a scalar (see below)
```
`get_atom14_mask(aatype) = RESTYPE_ATOM14_MASK[aatype]` — trivial gather,
`RESTYPE_ATOM14_MASK` already exists in the port
(`salad_torch/io/residue_constants.py`).

`violation_loss` internally calls `find_structural_violations` then, because
`per_residue=False`, `custom_structural_violation_loss` (both in
`alphafold_loss.py`). **`compute_violation_metrics`
(`salad/aflib/model/folding_multimer.py:291`) is also called but its result
is discarded (`violation, _ = violation_loss(...)`)** — do not port it, it
contributes nothing to the loss value and is pure wasted compute if
implemented.

`find_structural_violations` (`alphafold_loss.py:378-471`) computes 3
components, combined by `custom_structural_violation_loss`
(`alphafold_loss.py:569-584`):
```text
violation = (bonds_c_n_loss_mean
             + 0.3 * angles_ca_c_n_loss_mean
             + 0.3 * angles_c_n_ca_loss_mean
             + clashes_mean_loss
             + sum(within_residues_per_atom_loss_sum) / (sum(pred_mask) + 1e-6))
```
This is already a **single scalar** for the whole (flattened, multi-item)
batch — no `base_weight`, no per-item equalisation anywhere in this term;
`.mean()` on it in NSB:1430 is a literal no-op.

**(i) `between_residue_bond_loss`** (`all_atom_multimer.py:497-595`) — C-N
bond length + Cα-C-N and C-N-Cα bond angles between sequence-adjacent
residues:
```text
this_ca, this_c = pred_atom_pos[:-1, 1], pred_atom_pos[:-1, 2]      # slice OFF the last residue
next_n, next_ca = pred_atom_pos[1:, 0], pred_atom_pos[1:, 1]         # slice OFF the first residue
has_no_gap_mask = (residue_index[1:] - residue_index[:-1] == 1.0)    # float, 1.0 or 0.0

c_n_bond_length = ||this_c - next_n||_{eps=1e-6}                      # euclidean_distance == vec_norm(a-b, eps) with eps clamped INSIDE the sqrt (already ported: salad_torch/geometry/rigid.py `vec_norm`)
next_is_proline = (aatype[1:] == restype_order['P'])
gt_length = lerp(between_res_bond_length_c_n[0], [1], next_is_proline)     # [1.329, 1.341]
gt_stddev = lerp(between_res_bond_length_stddev_c_n[0], [1], next_is_proline)  # [0.014, 0.016]
c_n_error = sqrt(1e-6 + (c_n_bond_length - gt_length)**2)
c_n_loss_per_residue = relu(c_n_error - 2.0 * gt_stddev)               # tolerance_factor_soft = tolerance_factor_hard = c.violation_tolerance_factor = 2.0
m = this_c_mask * next_n_mask * has_no_gap_mask
c_n_loss = sum(m * c_n_loss_per_residue) / (sum(m) + 1e-6)              # SCALAR

c_ca_unit = normalized(this_ca - this_c, eps=1e-6)
c_n_unit  = (next_n - this_c) / c_n_bond_length                          # NOT renormalised via .normalized(), just divided by the (already eps-clamped) bond length computed above
n_ca_unit = normalized(next_ca - next_n, eps=1e-6)
ca_c_n_cos = dot(c_ca_unit, c_n_unit)
gt_angle, gt_stddev = between_res_cos_angles_ca_c_n            # [-0.4473, 0.0311]
ca_c_n_error = sqrt(1e-6 + (ca_c_n_cos - gt_angle)**2)
ca_c_n_loss_per_residue = relu(ca_c_n_error - 2.0*gt_stddev)
ca_c_n_loss = sum(mask' * ...) / (sum(mask')+1e-6)                       # mask' = this_ca_mask*this_c_mask*next_n_mask*has_no_gap_mask

c_n_ca_cos = dot(-c_n_unit, n_ca_unit)
gt_angle, gt_stddev = between_res_cos_angles_c_n_ca             # [-0.5203, 0.0353]
# identical relu/tolerance pattern -> c_n_ca_loss (SCALAR)

# per-residue diagnostic (feeds violation MASKS used by clash detection, not directly summed into the scalar loss):
per_residue_loss_sum = c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue    # (L-1,)
per_residue_loss_sum = 0.5 * (pad_right(per_residue_loss_sum, 1) + pad_left(per_residue_loss_sum, 1))  # (L,) split the edge loss half onto each neighbour
```
**Upstream trap #1 — no batch/chain gating at all.** `between_residue_bond_loss`
has no `batch_index` or `chain_index` parameter. Its only protection against
computing a fake "peptide bond" between the last residue of one item/chain
and the first residue of the next is the `has_no_gap_mask` check
(`residue_index` must increase by *exactly* `1.0`). This works in practice
only because `residue_index` is expected to reset (not continue
incrementing by 1) at every chain/batch-item boundary in salad's data
pipeline — it is an **implicit invariant**, not an explicit safety mask.
Preserve `residue_index` exactly as given (integer, per-chain local, always
non-monotonic across a boundary) — do not "clean it up" into a single
globally increasing index, or this loss will silently start scoring
cross-protein backbone geometry.

Constants needed (not yet in the port; add as literal tables to
`salad_torch/io/residue_constants.py`, values from
`salad/aflib/common/residue_constants.py:483-488`):
```text
between_res_bond_length_c_n        = [1.329, 1.341]
between_res_bond_length_stddev_c_n = [0.014, 0.016]
between_res_cos_angles_c_n_ca      = [-0.5203, 0.0353]
between_res_cos_angles_ca_c_n      = [-0.4473, 0.0311]
```

**(ii) `custom_clash_loss`** (`alphafold_loss.py:474-565` — this
**monkey-patches** `all_atom_multimer.between_residue_clash_loss` at import
time via `setattr`, and additionally supports `batch_index`, which the
un-patched AlphaFold original does not; this is the version actually
executed via `find_structural_violations`):
```text
residue_index = residue_index + 2000 * chain_index      # NOT residue_index += ...! JAX arrays are immutable so upstream's
                                                            # `residue_index += 2000*chain_index` silently creates a new array
                                                            # and rebinds the local name; a naive torch port using an in-place
                                                            # `residue_index += ...` would corrupt the CALLER's tensor (shared
                                                            # with everything else computed from `data["residue_index"]` this
                                                            # step). Always use out-of-place `+` here.
dists = ||pred_pos[:, None, :, None] - pred_pos[None, :, None, :]||_{eps=1e-10}     # (L, L, 14, 14)
dists_mask = atom_exists[:, None, :, None] * atom_exists[None, :, None, :]
dists_mask *= (residue_index[:, None, None, None] < residue_index[None, :, None, None])     # upper triangle only (avoid double count)
dists_mask *= (batch_index[:, None, None, None] == batch_index[None, :, None, None])          # explicit batch gate (added relative to AlphaFold's original; chain-safety here comes from the +2000*chain_index offset above making same-chain adjacency the only "adjacent" case, NOT from an explicit chain mask)
c_n_bonds = (residue_index[:,None,None,None]+1 == residue_index[None,:,None,None]) * onehot(2,14)[atom1] * onehot(0,14)[atom2]   # exclude the real peptide C-N bond from being scored as a clash
dists_mask *= (1 - c_n_bonds)
disulfide = onehot(SG_idx,14)[atom1] * onehot(SG_idx,14)[atom2]           # SG = Cys sulfur, atom14 index looked up from restype_name_to_atom14_names['CYS']
dists_mask *= (1 - disulfide)
lower_bound = dists_mask * (atom_radius[:,None,:,None] + atom_radius[None,:,None,:])     # sum of van-der-Waals radii
error = dists_mask * relu(lower_bound - 1.5 - dists)                       # overlap_tolerance_soft = clash_overlap_tolerance = 1.5
mean_loss = sum(error) / (1e-6 + sum(clash_mask))                          # clash_mask = dists_mask * (dists < lower_bound - 1.5)  [hard==soft==1.5 here]
# also returns per_atom_loss_sum / per_atom_clash_mask, both (L,14), consumed only by within_residues' sum below and by the (unused) per-residue violation mask
```
`atom_radius (L, 14)`: `van_der_waals_radius[atom_type[0]]` gathered per
atom14 slot via `get_atom14_to_atom37_map(aatype)`; only `{'C':1.7, 'N':1.55,
'O':1.52, 'S':1.8}` are ever looked up for standard amino acids (add these 4
literal values to the port's residue constants).

**(iii) `within_residue_violations`** (`all_atom_multimer.py:681-729`) —
purely intra-residue 14×14 atom-pair distance bounds, no cross-residue or
batch coupling at all (safe by construction):
```text
dists_mask = (1 - eye(14)) * atom_exists[:,:,None] * atom_exists[:,None,:]     # (L,14,14)
dists = ||pred_pos[:,:,None] - pred_pos[:,None,:]||_{eps=1e-10}
low_err  = relu(dists_lower_bound - dists)          # tighten_bounds_for_loss = 0.0
high_err = relu(dists - dists_upper_bound)
per_atom_loss_sum = sum( dists_mask * (low_err + high_err), axis=1) + sum(..., axis=2)    # (L, 14)
```
`dists_lower_bound`/`dists_upper_bound` `(21, 14, 14)`, per-restype, from
`make_atom14_dists_bounds(overlap_tolerance=1.5,
bond_length_tolerance_factor=2.0)` (`residue_constants.py:885-927`, called
with `violation_tolerance_factor=2.0` — note this function's OWN default
argument is `bond_length_tolerance_factor=15`, but salad always calls it
with the config's `violation_tolerance_factor=2.0`, overriding the default;
do not use the function's default). This table needs, for every
(restype, atom1, atom2) pair with `atom1 != atom2`:
- if no literature bond/virtual-bond exists between them: `lower =
  vdw(atom1)+vdw(atom2)-1.5`, `upper = 1e10` (a clash-avoidance floor with
  no ceiling);
- if they ARE bonded (or connected by exactly one intermediate atom, i.e. a
  bond angle — see "virtual bonds" below), that entry is **overwritten**:
  `lower = length - 2.0*stddev`, `upper = length + 2.0*stddev`.

The bonded/virtual-bonded values come from
`load_stereo_chemical_props()` (`residue_constants.py:392-478`), which
parses a **literal literature data table**,
`salad/aflib/common/stereo_chemical_props.txt` (346 lines: per-restype bond
lengths + stddevs, then per-restype bond angles + stddevs, converted to a
third "virtual bond" table via the law of cosines: `virtual_length =
sqrt(l1² + l2² - 2·l1·l2·cos(angle))` with the obvious error-propagation
formula for `virtual_stddev`, NSB-adjacent code at
`residue_constants.py:459-474`). **This is pure physical-constant data, not
model logic** — copy `stereo_chemical_props.txt` verbatim into the port
(e.g. `salad_torch/io/stereo_chemical_props.txt`) and port
`load_stereo_chemical_props` + `make_atom14_dists_bounds` as straight-line
table-building code; do not attempt to re-derive or approximate these
numbers, and do not treat copying this one data file as "reading the JAX
source" in the sense the writer is meant to avoid — there is no algorithm to
misunderstand here, only literature bond geometry.

---

## 8. Summary: dead terms for `default_vp`

Do not wire these into the port's default training config; implement them
behind the same config flags if/when a non-`default_vp` variant is ported:

- `aa_trajectory` (needs `c.aa_trajectory`)
- `distogram_trajectory` (needs `c.distogram_trajectory`)
- `soft_lddt` / `lddt` (needs `c.soft_lddt`)
- `x_late` gating on the `x` loss (needs `c.x_late`)
- `mask_atom14` gating on `pos_mask` (needs `c.mask_atom14`)
- Kabsch alignment on `pos`/`atom_pos` ground truth (needs `c.kabsch`) — §4
- `get_sigma_edm` / EDM (`c.diffusion_kind=="edm"`) and flow-matching
  (`c.diffusion_kind in ("flow","flow_scaled")`) noising branches — only the
  `vp`/`vpfixed` branch of `apply_diffusion` (§1.4) is exercised
- `preconditioning_scale_factors(...)["loss"]`'s `edm`/`edm_scaled`/`flow`
  branches — for `default_vp` this function always returns the literal `1`

## 9. Dependencies the writer must add (not covered by "already ported")

1. `index_kabsch`, `index_align` — §4 (needed even though dead, gated by a
   `c.kabsch` flag for parity with other configs).
2. `diffuse_atom_cloud`, `diffuse_sequence`, `position_std` (dead helper used
   only when `cloud_std is None`, i.e. never for `default_vp` — port it
   anyway since `diffuse_atom_cloud`'s signature exposes it) —
   `salad/modules/utils/diffusion.py`, §1.4.
3. Violation loss stack: `between_residue_bond_loss`, `custom_clash_loss`,
   `within_residue_violations`, `custom_structural_violation_loss`,
   `find_structural_violations`, `violation_loss` — §7.10 — plus the new
   residue-constant tables and `stereo_chemical_props.txt` data file listed
   there. `get_atom14_mask` is a one-line gather against the port's existing
   `RESTYPE_ATOM14_MASK`.
4. `get_angle_positions` (NSB:1435-1457) and the `AADecoder` (NSB:1497-1584)
   — technically model-forward components, not loss terms, but `result["aa"]`,
   `result["atom_pos"]`, `result["corrupt_aa"]` all originate here and §7.1 /
   §7.5-7.9 depend on their exact semantics (in particular: atom14 side
   chains are built from **ground-truth** aatype during training, and the
   ADM sequence-masking ratio is independently sampled inside the decoder,
   not taken from `data["t_seq"]`). If a separate model-forward spec exists,
   cross-check it agrees with the semantics described in §7.1 and §7.5.
