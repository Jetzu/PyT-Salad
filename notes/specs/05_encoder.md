# SALAD Encoder — complete specification

Source of truth: `salad/modules/noise_schedule_benchmark.py` (JAX, read-only,
never edit) for `Encoder`/`EncoderBlock`/`EncoderStack`/`encoder_pair_features`/
`Atom14Encoder`/`Update`/`EncoderUpdate`/`get_residual_gadgets`, and
`salad/modules/geometric/geometric.py` for `VectorLinear`/`vector_mean_norm`/
`Linear`/`MLP`. Config: `default_vp` (`salad_torch/config.py:30`) —
`augment_size=15`, `encoder_depth=2`, `local_size=128`, `pair_size=64`,
`num_neighbours=48`, `resi_dual=False`, `heads=8`, `key_size=32`, `factor=4`.
All four released configs (`default_vp`, `default_vp_scaled`,
`default_ve_scaled`, `multimotif_vp`) share this exact Encoder — none of them
touch `augment_size`, `encoder_depth`, `resi_dual`, or `equivariance`.

Manifest cross-check: every shape and scope name below was verified against
`grep '~encode' /hnvme/workspace/v181bb10-salad/default_vp-200k.txt`, which
has **exactly 59** `structure_diffusion/~encode/...` entries. All 59 are
accounted for in §6's table with **zero discrepancies** — every JAX line this
spec cites was cross-checked line-by-line against a manifest shape and the
count matches exactly (14 top-level leaves + 45 leaves under
`__layer_stack_no_state/encoder_block`, see §6.4 for the arithmetic).

Notation: `L` = total residues in the flattened batch. `A = 20` = `5 +
augment_size` (backbone `N,CA,C,O` + idealised pseudo-`CB`, plus 15 learned
pseudo-atoms) — the atom axis the Encoder actually operates on internally.
This is **not** the atom14 axis. `K = num_neighbours = 48`.

---

## 0. What this component is and why it matters for training only

`Encoder` builds `data["pos"]`, the `(L, 20, 3)` "ground truth" position
tensor that the diffusion trunk is trained to denoise (spec-03 §1.2). It is
called once, on the clean (t=0) structure, inside
`StructureDiffusion.encode` (NSB:162-221), which itself is called once inside
`StructureDiffusion.prepare_data` (NSB:106-160) — i.e. **per training
example, not per diffusion step**. It is never called at de-novo sampling
time (there is no atom14 ground truth to encode), which is why the port
could ship without it (`training/loop.py` stands in with a deterministic
placeholder — see §7 below for the exact spot to replace).

It has real trainable weights (59 leaf tensors, all present in every
released checkpoint) that are jointly optimised with the rest of the model,
so a parity-faithful *training* port must run the real network here, not the
placeholder. Getting it wrong silently corrupts the training target for
every position/rotation/FAPE/distogram loss that reads `data["pos"]`
(spec-03 §7), because those losses supervise the trunk against
*this network's output*, not against a fixed geometric quantity.

**Two upstream facts the writer must not miss:**

- **(a) The Encoder is sequence-agnostic.** `aa_gt` is read in `encode()`
  (NSB:169) only to be forwarded unchanged as the `seq` return value
  (NSB:220) — the amino-acid identity never enters `Encoder.prepare_features`
  or `EncoderBlock`. (Contrast with the dead `Atom14Encoder`, §5, which *does*
  condition on `aa`.)
- **(b) Only the augment atoms (indices 5:20) are ever modified.** The 5
  backbone/pseudo-CB atoms (N, CA, C, O, CB) pass through the entire Encoder
  **completely unchanged**, both at input and output — see §4 step 7. The
  Encoder's entire job is to *invent* 15 additional per-residue pseudo-atom
  positions from context; it never touches real backbone geometry.

---

## 1. Call site: `StructureDiffusion.encode` (NSB:162-221)

```
def encode(self, data):
    pos = data["pos"]                 # (L, ≥4, 3) — atom14-truncated, centered,
                                       # gap-filled positions from prepare_data (NSB:106-160)
    aatype = data["aa_gt"]            # (L,) int, 0..19 — UNUSED by Encoder itself, only forwarded
    c = self.config
    backbone = positions_to_ncacocb(pos)      # (L, 5, 3): N, CA, C, O, pseudo-CB
    pos = backbone
    # c.equivariance == "semi_equivariant": DEAD for all 4 released configs
    # (equivariance is never set in default_vp/_scaled/default_ve_scaled/multimotif_vp,
    #  DotDict.get returns None -> falsy). Do not implement this branch.
    data["pos"] = pos                 # NOTE: this dict write is later superseded —
                                       # see the "stale data['pos']" pitfall in §8.
    if c.augment_size > 0:            # TRUE for every released config (augment_size=15)
        _, augmented_pos = Encoder(c)(data)     # <-- THIS spec. `local` return is DISCARDED.
        pos = augmented_pos
    elif c.encode_atom14:             # DEAD, see §5 (Atom14Encoder)
        ...
    else:
        pos = backbone                # unreachable for released configs
    # equivariance inverse-augmentation: DEAD (same flag as above)
    seq = aatype
    return seq, pos                   # pos: (L, 20, 3) — this becomes data["pos"] in prepare_data
```

**Exact contract the writer must reproduce:**
- Input to `Encoder.__call__`: the `data` dict, with `data["pos"]` set to the
  **5-atom** `(L, 5, 3)` N/CA/C/O/pseudo-CB backbone (from
  `positions_to_ncacocb`, already ported at `salad_torch/geometry/frames.py:49`),
  plus `data["residue_index"]`, `data["chain_index"]` (already the
  batch-uniquified chain ids from `prepare_data`), `data["batch_index"]`,
  `data["mask"]` (all already present in `data` by this point in
  `prepare_data`/`prepare_batch`).
- Output consumed at the call site: **only the second tuple element**
  (`augmented_pos`, `(L, 20, 3)`). The first element (`local`, the encoder's
  final per-residue feature vector) is thrown away here — it still needs to
  be computed and still carries real gradient (it feeds the position-update
  MLP inside `Encoder.__call__`, §4 step 5), so do not skip computing it as
  an "unused" shortcut; only its *return* to the caller is a no-op.

This is exactly `training/loop.py`'s documented gap (module docstring,
`prepare_batch` NSB-parity comment): the placeholder there
(`extra = backbone5[:, 4:5, :].expand(-1, augment_size, -1)`, i.e. replicate
the pseudo-CB into all 15 augment slots) must be replaced by a real
`Encoder(config)(data)` call once this module exists. Concretely, in
`salad_torch/training/loop.py`, `prepare_batch` currently does:

```python
backbone5 = positions_to_ncacocb(atom_pos)
if augment_size > 0:
    extra = backbone5[:, 4:5, :].expand(-1, augment_size, -1)
    pos20 = torch.cat([backbone5, extra], dim=1)
```

This block must become (schematically — `prepare_batch` would need an
`encoder: Encoder | None` argument, constructed once by the caller and
passed through, matching how `model` is threaded elsewhere in that file):

```python
backbone5 = positions_to_ncacocb(atom_pos)
if augment_size > 0:
    enc_data = {"pos": backbone5, "residue_index": raw["residue_index"],
                "chain_index": chain, "batch_index": batch_index, "mask": mask}
    _, pos20 = encoder(enc_data)
else:
    pos20 = backbone5
```

`encoder`'s weights must be part of the same optimizer / EMA / checkpoint as
the rest of the model (they are one leaf under the same `structure_diffusion`
Haiku tree upstream) — do not treat it as a frozen preprocessing step.

---

## 2. `Encoder.prepare_features` (NSB:700-741) — building the 20-atom input

Reads `data["pos"]` (5, 3), `data["residue_index"]`, `data["chain_index"]`,
`data["batch_index"]`, `data["mask"]`.

```
1. positions = data["pos"]                                        # (L, 5, 3)
2. frames, local_positions = extract_aa_frames(positions)          # frames: (L,) rigid; local_positions: (L, 5, 3)
   [already ported: salad_torch/geometry/frames.py:24, extract_aa_frames]

3. augment = VectorLinear(augment_size=15, initializer="linear")(local_positions)
   # -> (L, 15, 3), see §3 for the exact op. NOT zero-initialised (VectorLinear's
   # own default init is "zeros", but this call site explicitly overrides it
   # with init_linear(), i.e. VarianceScaling(fan_in, scale=1.0) truncated-normal —
   # do NOT use VectorLinear's class default here).
4. augment = vector_mean_norm(augment)                              # (L, 15, 3), see §3.2

5. local_positions = concat(local_positions[:, :5], augment, axis=-2)  # (L, 20, 3), LOCAL frame
6. positions = frames[:, None].apply_to_point(local_positions)      # (L, 20, 3), GLOBAL frame
   # frames[:, None]: insert an axis before the trailing point axis so one
   # frame broadcasts over all 20 atoms — Rigid.__getitem__ already supports
   # this (salad_torch/geometry/rigid.py:122).

7. neighbours = get_spatial_neighbours(num_neighbours=48)(positions[:, 4], batch_index, mask)
   # already ported: salad_torch/geometry/neighbours.py's get_spatial_neighbours.
   # Anchor point is atom index 4 = the pseudo-CB (NOT an augment atom).
   # "item" arg = batch_index, NOT chain — neighbours may cross chain
   # boundaries within the same batch item (whole-complex context). This is
   # the Encoder's ONLY neighbour-selection stage: unlike the diffusion trunk's
   # extract_neighbours (index+spatial+random, 3 stages), the Encoder uses a
   # single spatial-only stage of size 48, no `neighbours=` seed passed in.

8. [REDUNDANT, upstream really does this] _, local_positions = extract_aa_frames(positions)
   # positions here is the (L, 20, 3) GLOBAL array from step 6. This recomputes
   # a `frames` (discarded, `_`) that is mathematically IDENTICAL to the one
   # from step 2 (positions[..., :3] i.e. N/CA/C are byte-for-byte unchanged by
   # steps 5-6, since apply_to_point(apply_inverse_to_point(x)) round-trips),
   # and a `local_positions` that reproduces the pre-step-5 local coordinates
   # for atoms 0-4 and the post-step-4 `augment` for atoms 5-19, up to fp32
   # round-trip error. It is SAFE to skip this recompute and reuse the
   # `local_positions` from step 5 directly instead — the writer does not need
   # to reproduce it for correctness, only be aware upstream computes it twice.
   dist = local_positions.norm()                                   # (L, 20) — vec_norm, salad_torch/geometry/rigid.py:42

9. local_features = concat(
       normalized(local_positions).reshape(L, 60),                 # vec_normalized, rigid.py:47 -> (L, 20*3)
       distance_rbf(dist, 0.0, 22.0, bins=16).reshape(L, 320),      # (L, 20*16); reuse salad_torch/geometry/features.py:distance_rbf
       log(dist + 1).reshape(L, 20),                                # NOT distance_rbf'd, plain log1p-style (log(dist+1), not log1p(dist))
       axis=-1)                                                     # total (L, 400)
10. local = MLP(size=local_size*4=512, out_size=local_size=128, depth=2,
                activation=gelu, bias=False, init="relu", final_init="linear")(local_features)
    # reuse salad_torch/modules/basic.py MLP: MLP(400, 512, out_size=128, depth=2,
    #        activation=gelu, bias=False, final_init="linear")
    # NOTE bias=False here — both internal Linears skip their bias (b is still
    # allocated per HkLinear convention, just never added).
11. positions = positions.to_array()  # no-op in torch (already a plain tensor)
12. local = LayerNorm(128)(local)     # HkLayerNorm, create_scale=True, create_offset=True

return local (L,128), positions (L,20,3), neighbours (L,48), resi, chain, batch, mask
```

`pair_mask = (batch[:,None]==batch[None,:]) * mask[:,None] * mask[None,:]`
computed at the very top of upstream's `prepare_features` (right after
reading `data["mask"]`) is **dead** — never returned, never referenced again
in this method (the actually-used `pair_mask` is built per-block, from
`neighbours`, inside `encoder_pair_features`, §4). Do not bother computing it.

---

## 3. `VectorLinear` and `vector_mean_norm` (`geometric.py:282-301`, `:358-365`)

### 3.1 `VectorLinear`

```python
class VectorLinear(Linear):
    def __init__(self, size=128, initializer="zeros", name="vector_linear"):
        super().__init__(size, False, 0, initializer, name)   # bias=False, bias_init=0

    def __call__(self, vec):                       # vec: Vec3Array or plain array, (..., A_in, 3)
        result = vec.to_array() if isinstance(vec, Vec3Array) else vec
        result = swapaxes(Linear.__call__(self, swapaxes(result, -1, -2)), -1, -2)
        return Vec3Array.from_array(result) if isinstance(vec, Vec3Array) else result
```

`Linear.__call__` (`basic.py:51-58`, JAX) is `w = get_parameter("w",
[data.shape[-1], size]); b = get_parameter("b", [size]); out =
einsum("...k,ki->...i", data, w); out += b if self.bias else out` (bias
**always allocated**, added only if `self.bias` is `True`). Since
`VectorLinear` hardcodes `bias=False`, **`b` is allocated in every checkpoint
(`vector_linear/b`, shape `(15,)`) but is never added to the output** — dead
weight kept only for checkpoint-shape parity (it never receives gradient
either, so after loading a checkpoint it will sit at whatever its
init value from training was, or 0.0 if this is a fresh random init; either
way it is inert). The port's own convention already documents this exact
pattern (`salad_torch/modules/basic.py` module docstring, "Bias quirk
reproduced from upstream") — reuse `HkLinear` with `bias=False` and this
falls out for free.

**Exact torch semantics for the swapaxes trick.** The whole point is that the
"feature" axis `Linear` contracts is the **atom axis**, not the xyz axis —
this is what makes it an *equivariant* mixing of the atom positions (a fixed
real-valued linear combination of vectors is still SE(3)-equivariant; mixing
the xyz axis would not be). Given `vec` of shape `(..., A_in, 3)` and a
weight `w` of shape `(A_in, A_out)` (Haiku's `(in, out)` layout — confirmed by
the manifest: `vector_linear/w` is `(5, 15)`, i.e. `A_in=5` [the fixed
backbone atom count fed in at prepare_features step 3, NOT `A_out=augment_size=15`]):

```python
def forward(self, vec: Tensor) -> Tensor:              # vec: (..., A_in, 3)
    out = torch.einsum("...ac,ao->...oc", vec, self.w)  # self.w: (A_in, A_out), Haiku (in,out) layout
    return out                                           # (..., A_out, 3); bias never applied
```

This is exactly equivalent to upstream's `swapaxes(-1,-2)` → contract last
axis with `w` via `einsum("...k,ki->...i", ...)` → `swapaxes(-1,-2)` back;
the single `einsum("...ac,ao->...oc", ...)` above does the same contraction
in one step without materialising the transposed intermediate, and is the
form to use in torch (do not literally transpose-then-`HkLinear`-then-
transpose unless matching the einsum exactly is more convenient for reusing
`HkLinear` verbatim — either is numerically identical, but if reusing
`HkLinear` directly, remember its `forward` does
`einsum("...k,ki->...i", x, w)`, so you'd call it on `vec.transpose(-1,-2)`
whose last dim is `A_in`, then transpose the `(..., 3, A_out)` result back to
`(..., A_out, 3)` — both routes are fine, but pick one and do not accidentally
swap which axis is contracted).

Only one instantiation of `VectorLinear` exists in the whole model:
`prepare_features` step 3, `VectorLinear(15, initializer=init_linear())`,
i.e. `A_in=5` (the fixed 5-atom backbone), `A_out=15` (`augment_size`). It is
**not** reused inside `EncoderBlock`/`EncoderStack`/`EncoderUpdate` or
anywhere in the final position update (§4) — those use plain scalar `MLP`s
whose *output* gets reshaped into `(..., 3)`, which is a different
(non-equivariant-by-construction, only zero-init-safe) idiom already used
throughout the port (e.g. `UpdatePositions` in `blocks.py:214`).

### 3.2 `vector_mean_norm` (`geometric.py:358-365`)

```python
def vector_mean_norm(vec: Vec3Array) -> Vec3Array:
    vec = vec.to_array()
    vec = vec - vec.mean(axis=-2, keepdims=True)                      # center over the ATOM axis
    vec /= sqrt(maximum((vec**2).sum(axis=-1, keepdims=True), 1e-6)) \
             .mean(axis=-2, keepdims=True)                             # divide by mean per-atom norm
    return Vec3Array.from_array(vec)
```

Read this carefully — it is **not** per-atom unit normalization. It is:
(1) subtract the centroid across the atom axis (so the `A_out` augment atoms
are re-centered to have zero mean position, in whatever frame they currently
live in), THEN (2) compute each atom's distance from that new centroid
(`sqrt(clamp(sum_xyz(x²), min=1e-6))`, shape `(..., A, 1)`), THEN (3) average
those per-atom distances over the atom axis (shape `(..., 1, 1)`), THEN (4)
divide the *centered* cloud by that single scalar-per-residue mean distance.
Net effect: the `A` output vectors keep their relative directions/spread
pattern but are recentred to mean-zero and globally rescaled so their
*average* distance from that new center is 1. This is called twice in the
whole Encoder: once on the freshly-projected augment atoms
(`prepare_features` step 4, over `A=15`) and once on the final
after-refinement augment atoms (`Encoder.__call__` step 6 below, also over
`A=15` — never over all 20).

Exact torch implementation (a new function, not currently in the port; put
it in `salad_torch/geometry/rigid.py` next to `vec_normalized`, matching that
file's `_NORM_EPS = 1e-6` convention, or reuse that constant explicitly):

```python
def vector_mean_norm(vec: Tensor, eps: float = 1e-6) -> Tensor:  # vec: (..., A, 3)
    vec = vec - vec.mean(dim=-2, keepdim=True)
    per_atom_norm = torch.sqrt(torch.clamp((vec ** 2).sum(dim=-1, keepdim=True), min=eps))
    scale = per_atom_norm.mean(dim=-2, keepdim=True)
    return vec / scale
```

Note the clamp is `min=eps` on the **sum of squares**, i.e. `min=1e-6` on the
squared quantity directly (not `eps**2`) — this differs from the port's
`vec_norm`/`_NORM_EPS` convention (`vec_norm` clamps at `eps*eps` where
`eps=1e-6`, i.e. effectively `1e-12` on the squared sum). Upstream's
`vector_mean_norm` uses `jnp.maximum(..., 1e-6)` directly on the squared sum,
i.e. a **1e6× larger floor** than `vec_norm`'s. Do not substitute
`vec_norm`/`vec_normalized` here — implement the `1e-6`-on-squared-sum clamp
literally as shown, it is a different (and much less aggressive near-zero)
epsilon than the rest of the port uses.

---

## 4. `Encoder.__call__` (NSB:745-773) — full forward

```
local, pos, neighbours, resi, chain, batch, mask = self.prepare_features(data)   # §2
                                                                                   # pos: (L, 20, 3), GLOBAL frame

local = EncoderStack(config, depth=encoder_depth=2)(local, pos, neighbours, resi, chain, batch, mask)  # §5
                                                                                   # local: (L, 128); NOTE: pos is NOT
                                                                                   # updated by the stack — every block
                                                                                   # reads the SAME fixed `pos` (unlike
                                                                                   # the diffusion trunk, which updates
                                                                                   # positions every block).

frames, local_positions = extract_aa_frames(pos)          # pos: the ORIGINAL (L,20,3) from prepare_features,
                                                            # i.e. frames == the frames from prepare_features
                                                            # step 2/8 (identical, since N/CA/C never changed)
augment = local_positions[:, 5:]                           # (L, 15, 3) — the ORIGINAL (pre-refinement) local
                                                            # augment coordinates, straight from prepare_features

update = MLP(size=2*local_size=256, out_size=augment_size*3=45, depth=2,
             activation=gelu, bias=True, init="relu", final_init="linear")(local)
       # reuse MLP(128, 256, out_size=45, depth=2, activation=gelu, final_init="linear")
       # NOTE bias=True here (upstream does NOT pass bias=False, unlike step 10 of §2) —
       # both internal Linears of THIS MLP apply their bias. Also NOTE final_init is
       # passed as init_linear() (an object), which is behaviourally identical to the
       # MLP default string "linear" (both -> VarianceScaling(fan_in, scale=1.0)); it is
       # NOT zero-init, unlike almost every other "final" projection in the port
       # (UpdatePositions, Update.linear_out, etc. all use init_zeros()). This one does not.
update = update.reshape(L, 15, 3)

augment = augment + update                    # NOT in-place: JAX `+=` on an immutable array is
                                                # ordinary addition; use `augment = augment + update`
                                                # in torch too (no aliasing concern either way here,
                                                # but match the port's existing "never mutate a
                                                # possibly-shared tensor in place" convention)

local_positions = local_positions[:, :5]       # backbone/pseudo-CB, UNCHANGED, straight through
augment = vector_mean_norm(augment)            # §3.2, re-normalize the REFINED augment cloud

local_positions = concat(local_positions, augment, axis=-2)      # (L, 20, 3), LOCAL frame
pos = frames[:, None].apply_to_point(local_positions)             # (L, 20, 3), GLOBAL frame — SAME
                                                                    # frames object as used above (and
                                                                    # identical to prepare_features's)

return local, pos    # local: (L, 128) — DISCARDED by the caller (§1); pos: (L, 20, 3) — the real output
```

Restating the invariant from §0(b) precisely: atoms 0-4 (N, CA, C, O,
pseudo-CB) in the returned `pos` are **bit-for-bit** (up to the one
frame-apply/frame-apply-inverse round trip in fp32) the same 5 backbone
points passed into `Encoder.__call__` via `data["pos"]`
(`positions_to_ncacocb` output). Only atoms 5-19 differ from a naive
pass-through, and they differ by a learned, context-dependent refinement of
the single linear projection computed once in `prepare_features`.

---

## 5. `EncoderBlock` (NSB:816-838) and `EncoderStack` (NSB:848-870)

### 5.1 `EncoderBlock.__call__(features, pos, neighbours, resi, chain, batch, mask)`

```
residual_update, residual_input, _ = get_residual_gadgets(features, c.resi_dual)
# c.resi_dual == False for ALL released configs -> residual_update = prenorm_skip
# (lambda local, out: local + out), residual_input = prenorm_input
# (lambda local: LayerNorm(local)). The resi_dual==True branch (ResiDual,
# a tuple-valued `features`) is UNIMPLEMENTED anywhere else in this port
# (DiffusionBlock/AADecoderBlock also hardcode the plain prenorm-skip form) —
# do the same here: hardcode prenorm-skip, do not build a generic dispatch,
# and do not attempt resi_dual=True support.

pair, pair_mask = encoder_pair_features(config)(pos, neighbours, resi, chain, batch, mask)   # §5.3

features = features + SparseStructureAttention(config, local_size=128, pair_size=64,
                                                normalize=True)(
                            LayerNorm(128)(features), pos, pair, pair_mask, neighbours)
    # reuse salad_torch/modules/attention.py:SparseStructureAttention verbatim.
    # `pos` passed UNSCALED (no /sigma_data) — matching AADecoderBlock's convention,
    # NOT DiffusionBlock's (which divides by sigma_data). The Encoder has no notion
    # of a diffusion noise scale at all.

features = features + EncoderUpdate(config, local_size=128, atom_count=20)(
                            LayerNorm(128)(features), pos, chain, batch, mask)
    # EncoderUpdate is ALREADY IMPLEMENTED verbatim at salad_torch/modules/blocks.py:191
    # (== upstream's `Update(c, global_update=False)` called with `condition=None`,
    # confirmed against the manifest in §6.3 below — do not reimplement, just call it
    # with atom_count=20 (= 5 + augment_size, matching salad_torch/modules/diffusion.py:50's
    # existing `self.atom_count = 14 if c.encode_atom14 else 5 + c.augment_size` expression).

return features
```

Two independent `LayerNorm(128)` instances are created here (one feeding
attention, one feeding the update) — see §6 for exact Haiku naming
(`layer_norm_1`, `layer_norm_2` at the `encoder_block` scope).

### 5.2 `EncoderStack.__call__(local, pos, neighbours, resi, chain, batch, mask)` (default_name="self_cond_stack")

```
depth = encoder_depth = 2 (config default; never overridden by any released config)
stack of `depth` EncoderBlocks, EACH RECEIVING THE SAME (pos, neighbours, resi, chain, batch, mask)
  — only `features`/`local` threads through the stack; `pos` is fixed for the
  whole stack (contrast with DiffusionStack, where pos updates every block).
c.resi_dual == False for all released configs:
    local = LayerNorm(128)(stack_output)          # single LayerNorm AFTER the whole stack
# (the resi_dual==True branch — incremental accumulation + a different final
# LayerNorm combination — is DEAD for all released configs; do not implement it,
# same reasoning as §5.1.)
return local             # NOTE: EncoderStack returns ONLY `local`, not `pos` —
                          # Encoder.__call__ re-derives pos-related quantities
                          # itself in step 3 of §4, from the ORIGINAL prepare_features pos.
```

Implementation pattern: follow `AADecoderStack`/`DiffusionStack`
(`salad_torch/modules/decoders.py:127`, `salad_torch/modules/blocks.py:284`)
— an `nn.ModuleList` of `depth` independently-parametrized `EncoderBlock`
instances (NOT weight-shared), because upstream's `hk.experimental.layer_stack`
/ `block_stack(depth, block_size=1, with_state=False)` (NSB's
`block_stack`, already referenced in spec-03/04) stacks *independent* weights
along a leading axis of size `depth` and traces the block function once — the
checkpoint converter slices `[i]` off that leading axis exactly as the
existing `diffusion_block_map`-style converters already do for
`DiffusionStack`/`AADecoderStack`. This is confirmed directly by the
manifest: every parameter under `.../self_cond_stack/__layer_stack_no_state/
encoder_block/...` carries a leading shape-`2` axis (e.g.
`encoder_block/linear/w` is `(2, 66, 64)`, not `(66, 64)`).

`hk.remat` (gradient checkpointing) wraps each block upstream
(`stack_inner`, NSB:855-870) — this is a pure compute/memory tradeoff with no
numerical effect; the port may use `torch.utils.checkpoint` per block for
parity of memory behaviour, but it is not required for correctness and no
other ported stack in this codebase bothers with it (`DiffusionStack`/
`AADecoderStack` are plain Python loops) — match that precedent and skip it
unless a later memory-budget spec asks for it.

### 5.3 `encoder_pair_features(config)` (NSB:1845-1866)

Structurally identical to the already-ported `AADecoderPairFeatures`
(`salad_torch/modules/decoders.py:49-100`) **minus the amino-acid embedding
term** — i.e. it is `aa_decoder_pair_features` with the
`aa_embedding[neighbours]` addition and its associated `linear_aa`/
`layer_norm_aa` deleted, everything else byte-identical (same feature
functions, same order, same `pair_size`, same final `LayerNorm`+`MLP`):

```python
class EncoderPairFeatures(nn.Module):
    def __init__(self, config, atom_count: int = 20):
        p = config.pair_size                                   # 64
        self.linear_relpos = HkLinear(66, p, bias=False)                  # linear
        self.linear_dist   = HkLinear(400, p, bias=False)                 # linear_1
        self.linear_dir    = HkLinear(atom_count * 3, p, bias=False)      # linear_2  (60 for atom_count=20)
        self.linear_rot    = HkLinear(9, p, bias=False)                   # linear_3
        self.linear_pairvec = HkLinear(12 * atom_count, p, bias=False)    # linear_4  (240 for atom_count=20)
        self.layer_norm = HkLayerNorm(p)
        self.mlp = MLP(p, p * 2, out_size=p, depth=2, activation=gelu, final_init="linear")

    def forward(self, pos, neighbours, resi, chain, batch, mask):
        pair_mask = mask[:, None] & mask[neighbours] & (neighbours != -1)
        pair = self.linear_relpos(GF.sequence_relative_position(32, one_hot=True)(
                    resi, chain, batch, neighbours))
        pair = pair + self.linear_dist(GF.distance_features(pos, neighbours, 0.0, 22.0))
        pair = pair + self.linear_dir(GF.direction_features(pos, neighbours))
        pair = pair + self.linear_rot(GF.position_rotation_features(pos, neighbours))
        pair = pair + self.linear_pairvec(GF.pair_vector_features(pos, neighbours))
        pair = self.layer_norm(pair)
        pair = self.mlp(pair)
        return pair, pair_mask
```

Every one of `GF.sequence_relative_position`/`distance_features`/
`direction_features`/`position_rotation_features`/`pair_vector_features` is
already ported verbatim (`salad_torch/geometry/features.py`) and already
exercised by `AADecoderPairFeatures`/`DiffusionPairFeatures` with these exact
same call signatures — no new geometry code is needed here, only this one
new `nn.Module` wrapper (or reuse the `AADecoderPairFeatures` class with a
flag to skip its `aa` term, whichever fits the codebase's taste better; the
Haiku scope name is what must match, see §6, not the Python class identity).
`atom_count=20` for the Encoder specifically (`direction_features`/
`pair_vector_features` scale with it; `distance_features` and
`position_rotation_features` are fixed-size regardless of `atom_count`,
confirmed by the manifest: `linear_1`/`linear_3` are `400`/`9` wide exactly
as in every other pair-feature builder in this codebase).

`pair_mask` here also gets recomputed as `pair, pair_mask` inside
`EncoderBlock` before being handed to `SparseStructureAttention` — no
`make_pair_mask` helper is invoked (unlike `DiffusionPairFeatures`, this
inline `mask[:,None] & mask[neighbours] & (neighbours != -1)` pattern is used
directly, matching the already-ported `AADecoderPairFeatures`/
`DistogramDecoder` convention exactly).

---

## 6. Haiku scope names and shapes — full manifest cross-check

Top-level path prefix for everything below:
`structure_diffusion/~encode/encoder/...` The `~encode` segment is Haiku's
automatic scoping of `StructureDiffusion.encode` (a non-`__call__` method —
Haiku auto-wraps parameter-creating calls inside any hk.Module method other
than `__call__` in a scope named `~<method_name>`, unless the method is
`@hk.transparent`; `encode` is not). `encoder` is `Encoder`'s own module name
(created via `__call__`, so no tilde). `~prepare_features` is likewise
`Encoder.prepare_features`'s auto-scope (again a non-`__call__` method).

### 6.1 `~prepare_features` scope (8 leaves, §2 steps 3-12)

| Haiku path | shape | role |
|---|---|---|
| `~prepare_features/vector_linear/w` | `(5, 15)` | `VectorLinear` weight (§3.1) — **in=5** (backbone atom count), **out=15** (`augment_size`) |
| `~prepare_features/vector_linear/b` | `(15,)` | allocated, **never applied** (§3.1) |
| `~prepare_features/mlp/linear/w` | `(400, 512)` | local-features MLP, hidden layer (in=400 = 60+320+20, §2 step 9-10) |
| `~prepare_features/mlp/linear/b` | `(512,)` | allocated, **never applied** (`bias=False` on this `MLP`) |
| `~prepare_features/mlp/linear_1/w` | `(512, 128)` | local-features MLP, output layer |
| `~prepare_features/mlp/linear_1/b` | `(128,)` | allocated, **never applied** |
| `~prepare_features/layer_norm/scale` | `(128,)` | final LayerNorm on `local` (§2 step 12) |
| `~prepare_features/layer_norm/offset` | `(128,)` | ditto |

### 6.2 `self_cond_stack` scope, outside the layer-stack (2 leaves)

| Haiku path | shape | role |
|---|---|---|
| `self_cond_stack/layer_norm/scale` | `(128,)` | final LayerNorm after the whole `EncoderStack` (§5.2) |
| `self_cond_stack/layer_norm/offset` | `(128,)` | ditto |

### 6.3 `self_cond_stack/__layer_stack_no_state/encoder_block/...` scope (45 leaves, leading axis = `encoder_depth=2`)

Everything here has an extra leading dim of size 2 (one slice per stack
depth); the table gives the per-layer shape (drop the leading `2` to get a
single `EncoderBlock`'s own parameter shape).

`encoder_pair_features` (a plain function upstream, so its Linears/LayerNorm/
MLP land flat in the `encoder_block` scope, sharing its counter — exactly the
pattern already documented for `aa_decoder_pair_features` in
`salad_torch/modules/decoders.py`'s module docstring):

| Haiku path (leading `2,` omitted) | shape | role |
|---|---|---|
| `linear/w`, `linear/b` | `(66,64)`,`(64,)` | `linear_relpos` |
| `linear_1/w`, `linear_1/b` | `(400,64)`,`(64,)` | `linear_dist` |
| `linear_2/w`, `linear_2/b` | `(60,64)`,`(64,)` | `linear_dir` (atom_count=20 → 60) |
| `linear_3/w`, `linear_3/b` | `(9,64)`,`(64,)` | `linear_rot` |
| `linear_4/w`, `linear_4/b` | `(240,64)`,`(64,)` | `linear_pairvec` (atom_count=20 → 240) |
| `mlp/linear/{w,b}` | `(64,128)`,`(128,)` | pair-features MLP hidden |
| `mlp/linear_1/{w,b}` | `(128,64)`,`(64,)` | pair-features MLP out |

Then, directly in `EncoderBlock.__call__` (§5.1), in program order:

| Haiku path | shape | role |
|---|---|---|
| `layer_norm/{scale,offset}` | `(64,)` each | the pair-features' OWN final LayerNorm (created inside `encoder_pair_features`, but it's the first bare `LayerNorm` in this scope so it gets the unsuffixed name — same flat-namespace effect as the linears above) |
| `sparse_structure_attn/...` | — | `SparseStructureAttention` (§5.1) — **already ported**, `salad_torch/modules/attention.py`; 15 leaves: `ada_point_attention/{layer_norm/{scale,offset}, qkv/{w,b}, layer_norm_1/{scale,offset}, layer_norm_2/{scale,offset}, qkv_global/linear/{w,b}, bias/{w,b}, gamma, project_out/{w,b}}`. `gamma` shape `(8,)` = `heads`; `layer_norm_1`/`layer_norm_2` shape `(32,)` = `key_size`; confirms `heads=8, key_size=32, normalize=True` exactly as `config` specifies. |
| `layer_norm_1/{scale,offset}` | `(128,)` each | `residual_input(features)` before the attention call (§5.1) — the SECOND bare LayerNorm created directly in `encoder_block` scope |
| `light_global_update/...` | — | `EncoderUpdate` (§5.1) — **already ported**, `salad_torch/modules/blocks.py:191`; 10 leaves: `mlp/linear/{w,b}` `(60,256)`/`(256,)`, `mlp/linear_1/{w,b}` `(256,128)`/`(128,)`, `linear/{w,b}` `(128,512)`/`(512,)`, `linear_1/{w,b}` `(128,512)`/`(512,)`, `linear_2/{w,b}` `(512,128)`/`(128,)`. (`local_size*factor = 128*4 = 512`.) |
| `layer_norm_2/{scale,offset}` | `(128,)` each | `residual_input(features)` before the `EncoderUpdate` call — the THIRD bare LayerNorm in `encoder_block` scope |

Arithmetic check: pair-features (10+4+2=16) + `sparse_structure_attn` (15) +
`layer_norm_1`+`layer_norm_2` (4) + `light_global_update` (10) = **45**,
matching the manifest exactly (lines 194-238 of the grep, inclusive).

### 6.4 `encoder` scope, directly (4 leaves, §4 "update" MLP)

| Haiku path | shape | role |
|---|---|---|
| `encoder/mlp/linear/w` | `(128, 256)` | position-update MLP hidden (in=`local_size`=128, out=`2*local_size`=256) |
| `encoder/mlp/linear/b` | `(256,)` | **applied** (this MLP has `bias=True`, unlike §6.1's) |
| `encoder/mlp/linear_1/w` | `(256, 45)` | position-update MLP out (out=`augment_size*3`=45) |
| `encoder/mlp/linear_1/b` | `(45,)` | **applied** |

Total: 8 (§6.1) + 2 (§6.2) + 45 (§6.3) + 4 (§6.4) = **59**, matching the
manifest count exactly. No discrepancies were found anywhere in this
cross-check.

---

## 7. `Atom14Encoder` (NSB:776-807) — DEAD for every released config

`c.encode_atom14` is never set truthy in `default_vp`, `default_vp_scaled`,
`default_ve_scaled`, or `multimotif_vp` (`DotDict.__getattr__ = dict.get`, so
an unset key reads `None` → falsy), and the class's own docstring says
"EXPERIMENTAL: Not used in the salad manuscript." **This module is
unreachable from `StructureDiffusion.encode` for every checkpoint that
exists**, and gates two further flags (`atom14_ca`, `atom14_learned`) that
are also unset. The writer should **not implement it** — there is nothing in
any released checkpoint to load into it, and it would add an
untested/unreachable code path. For completeness, if a future config ever
needs it: it takes atom14 positions + `aa_gt` (unlike `Encoder`, this variant
*is* sequence-conditioned), builds `local_features` from
`distance_rbf`+direction-normalized atom14 local coords + a 20-way one-hot
`aa`, and regresses all 14 atom positions directly via a single `MLP(local_size*2,
3*14, bias=True, final_init="linear")` — no attention stack, no neighbours,
no iterative refinement, structurally much simpler than `Encoder`. Do not
build this now.

---

## 8. Config flags — what gates what for `default_vp`

| flag | default_vp value | effect |
|---|---|---|
| `augment_size` | 15 | **ACTIVE.** `>0` → `Encoder` path taken in `encode()`. Also sizes `VectorLinear`'s output and the final position-update MLP's output (`augment_size*3=45`). |
| `encoder_depth` | 2 | **ACTIVE.** `EncoderStack` depth; also the leading axis on all `encoder_block`-scoped params (§6.3). |
| `local_size` | 128 | **ACTIVE.** Encoder's own local feature width (independent of the trunk's `local_size`, though they're numerically equal for every released config — do not assume they must be). |
| `pair_size` | 64 | **ACTIVE.** Encoder's pair feature width. |
| `num_neighbours` | 48 | **ACTIVE.** Sole neighbour-selection parameter for the Encoder — a single `get_spatial_neighbours` stage, unlike the trunk's 3-stage `extract_neighbours`. |
| `resi_dual` | False | **Gates dead code.** `True` branch (ResiDual residual style, both in `EncoderBlock` and `EncoderStack`) is unimplemented anywhere in this port; hardcode the `False` (prenorm-skip) behaviour, do not build a dispatch. |
| `equivariance` | unset (`None`) | **Fully dead.** Both `semi_equivariant` branches that would otherwise wrap `encode()`'s Encoder call in data-augmentation are skipped entirely. |
| `encode_atom14`, `atom14_ca`, `atom14_learned` | all unset | **Fully dead**, see §7. |
| `heads`, `key_size`, `factor` | 8, 32, 4 | **ACTIVE**, consumed by `SparseStructureAttention`/`EncoderUpdate` exactly as the trunk consumes them — no Encoder-specific override exists. |

---

## 9. Summary of subtleties for the writer (checklist)

1. **Bias on/off per layer, exactly:** every `Linear` inside
   `encoder_pair_features`/`~prepare_features`'s first MLP has `bias=False`
   (but `b` is still allocated in the checkpoint, per §3.1's Haiku
   convention — always allocate, conditionally apply, matching
   `salad_torch/modules/basic.py`'s existing `HkLinear`). The **one**
   exception is `Encoder.__call__`'s position-update `MLP`
   (§4/§6.4, `encoder/mlp`), whose two Linears both have `bias=True` and
   really apply it. `VectorLinear` always has `bias=False` (hardcoded,
   §3.1).
2. **Two different epsilon/clamp conventions coexist**: `vec_norm`/
   `vec_normalized` (`_NORM_EPS=1e-6`, clamped as `eps**2=1e-12` on the
   squared sum) vs. `vector_mean_norm` (clamped as `1e-6` directly on the
   squared sum, i.e. 1e6× looser) — §3.2. Do not conflate them.
3. **`pos` is fixed across the whole `EncoderStack`** — only `local`/
   features get updated block-to-block; contrast with `DiffusionStack`
   where `pos` is the thing that evolves. Passing an updated `pos` into
   later blocks would silently diverge from upstream.
4. **Only atoms 5:20 (the augment slots) are ever modified**; atoms 0:5
   (backbone + pseudo-CB) are a pure pass-through, unchanged from
   `Encoder.__call__`'s input to its output (§0b, §4).
5. **`Encoder`'s `local` return value is discarded** by `StructureDiffusion.encode`
   (§1) — still must be computed (it's the direct input to the position
   update MLP), just not propagated further.
6. **The final position-update MLP's `final_init` is NOT zero-init**
   (`init_linear()`, i.e. `VarianceScaling(fan_in, scale=1.0)`) — unusual for
   a "delta" projection in this codebase (contrast `UpdatePositions`,
   `Update.linear_out`, both zero-init). This means the Encoder's augment
   positions are **not** a small perturbation of the initial `VectorLinear`
   projection at initialization; they can move by an $O(1)$ amount from
   step 0 of training. Do not "fix" this to zero-init by analogy with the
   rest of the codebase — reproduce upstream exactly.
7. **The redundant second `extract_aa_frames` call in `prepare_features`**
   (§2 step 8) is safe to elide (mathematically a no-op round trip); do not
   spend effort reproducing the exact float bit pattern of doing it twice.
8. **Stale `data["pos"]` write in `encode()`** (§1): `encode()` sets
   `data["pos"] = pos` (the 5-atom backbone) right before calling `Encoder`,
   but the dict is never read again inside `encode()`/`prepare_data`'s
   return — `prepare_data`'s final returned dict is built fresh from local
   variables (`pos=pos` where `pos` is by-then the 20-atom `augmented_pos`).
   If the writer's `prepare_batch`/equivalent builds its output dict
   incrementally by mutating a shared `data` dict rather than from local
   variables, take care not to leave the stale 5-atom write as the final
   value of a `"pos"` key that downstream code expects to be 20-atom.
9. **`atom_count=20`** (`5+augment_size`) is the value to use for
   `EncoderPairFeatures`/`EncoderUpdate`'s `atom_count` argument — matching
   the existing expression at `salad_torch/modules/diffusion.py:50`. Do not
   confuse with the atom14 axis (14) or the backbone-only axis (5).
10. **Neighbour selection crosses chains.** `get_spatial_neighbours` here is
    called with `item=batch_index`, not `chain` — a residue's 48 spatial
    neighbours may belong to a different chain within the same complex.
    This is deliberate (the Encoder needs whole-complex context to place
    pseudo-atoms sensibly for interfaces) and differs from the (also
    batch-only, as it happens) trunk convention only in that some *other*
    trunk neighbour stages (`get_index_neighbours`) restrict to same-chain
    — the Encoder has no such stage at all.
