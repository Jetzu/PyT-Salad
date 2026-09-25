# SALAD denoiser architecture — PyTorch port spec

Main file: `salad/modules/noise_schedule_benchmark.py` (2858 lines, "**NSB**" below).
All haiku scopes and shapes below were verified against the real checkpoints.

## 0. Orientation

- `default_vp` is **not a class** — it's a `dotdict` config at
  `salad/modules/config/noise_schedule_benchmark.py:3`. The module is
  `StructureDiffusion` (training) / `StructureDiffusionInference` /
  `StructureDiffusionPredict` (inference), all with haiku name
  `"structure_diffusion"`, so **one checkpoint serves all**. The real network is
  the child module `Diffusion` (NSB:990).
- `dotdict.__getattr__ = dict.get` ⇒ **absent keys return `None`**. About half the
  branches in NSB are gated on keys that simply don't exist in `default_vp`.
- **The `Encoder` is never called for de-novo generation.** Its ~1.2 M params
  (`~encode/encoder/*`) are dead weight unless doing motif scaffolding or
  partial diffusion from a template.

| checkpoint | params | groups |
|---|---|---|
| default_vp / default_vp_scaled | 11,538,916 | 132 (bit-identical trees) |
| default_ve_scaled | 11,854,564 | 136 |
| multimotif_vp | 11,856,100 | 138 |

## 1. Call graph (default_vp, de-novo)

```
StructureDiffusionPredict.__call__                             NSB:645
├─ prepare_condition(data)                                     NSB:482
│   └─ Condition(c)(...)                                       NSB:2521
│       └─ 3 x [Linear -> LayerNorm]   aa 21->128, dssp 3->128, dssp_mean 3->128
│   └─ pair_condition = dict(dmap (N,N), omap (N,N,9),
│                            dmap_mask (N,N) bool, flags (N,N,3))
└─ Diffusion(c)(data, prev, predict_aa=True)                   NSB:997
    ├─ prepare_features(data, prev)                            NSB:1128
    │   └─ MLP(1725 -> 512 -> 128) -> LayerNorm
    ├─ get_random_neighbours(fape_neighbours=64)               (aux only, burns a key)
    ├─ DiffusionStack(c, DiffusionBlock)                       NSB:1587
    │   [EDMDiffusionStack NSB:1688 iff preconditioning=="edm_scaled"]
    │   └─ block_stack(depth=6, block_size=1, with_state=True) -> hk.scan
    │       └─ DiffusionBlock.__call__                         NSB:2172
    │           ├─ extract_neighbours(16,16,32)(pos)            -> (N,64)
    │           ├─ extract_condition_neighbours(16,16,16,16)(prev_pos) -> (N,64)
    │           ├─ diffusion_pair_features(c)(pos, ...)        NSB:2034
    │           ├─ SparseStructureAttention -> SparseInvariantPointAttention
    │           ├─ diffusion_pair_features(c)(prev_pos, ...)
    │           ├─ SparseStructureAttention   (2nd instance)
    │           ├─ Update(c, global_update=True)               NSB:2662
    │           └─ update_positions(pos, local_norm, out_scale) NSB:2740
    ├─ AADecoder(c).train(...)                                 NSB:1497/1548
    │   └─ 3 x AADecoderBlock (attention + EncoderUpdate)
    ├─ dssp_decoder = Linear(3, zeros)                         NSB:1033   (aux)
    ├─ DistogramDecoder(c)(...)                                NSB:1460   (aux)
    └─ get_angle_positions(...)                                NSB:1435
        -> MLP(529 -> 256 -> 14) -> 7x2 torsions -> single_protein_sidechains
```

## 2. Parameter map

`SD = structure_diffusion`. Haiku `Linear` stores `w [in,out]` — **transpose vs
`torch.nn.Linear.weight`**. `Linear.__call__` (`basic.py:51`) **always creates
`b` even when `bias=False`**, so ~40 % of the `b` entries in the checkpoint are
never added. `hk.LayerNorm([-1],True,True)` → `scale`, `offset`, eps 1e-5.

### 2.1 Trunk — `SD/diffusion/diffusion_stack/__layer_stack_with_state/diffusion_block/*`

Every param carries a **leading axis of 6** (`diffusion_depth`). Per layer
(pair_size 64, local_size 128, A = 5+augment_size = 20):

| scope | shape | role |
|---|---|---|
| `linear` | (66,64) | relpos one-hot, 2·32+2 bins |
| `linear_1` | (64,64) | `distance_rbf(dmap,0,22,64)`, masked by `dmap_mask` |
| `linear_2` | (3,64) | `flags` (chain_contact, rel_hotspot, block_contact) |
| `linear_3` | (400,64) | `distance_features` 5×5×16 RBF |
| `linear_4` | (60,64) | `direction_features` A×3 |
| `linear_5` | (9,64) | `position_rotation_features` |
| `linear_6` | (240,64) | `pair_vector_features` 2A×3 dirs + 2A×3 coords |
| `layer_norm` | 64 | pair LN |
| `mlp/linear`, `mlp/linear_1` | (64,128),(128,64) | pair MLP, gelu |
| `layer_norm_1` | 128 | prenorm before attn #1 |
| `sparse_structure_attn/…` | §2.2 | IPA over current neighbours |
| `linear_7 … linear_13` | 66,64,3,400,60,9,240 → 64 | 2nd pair block (over `prev_pos`) |
| `layer_norm_2`, `mlp_1/*`, `layer_norm_3` | | 2nd pair MLP + prenorm |
| `sparse_structure_attn_1/…` | §2.2 | IPA over condition neighbours |
| `layer_norm_4` | 128 | prenorm before `Update` |
| `light_global_update/linear` | (128,128) | condition embed, zeros, no bias |
| `light_global_update/mlp/*` | (60,256),(256,128) | local coords → gelu MLP, zeros |
| `light_global_update/linear_1` | (128,512) | local_update (factor 4) |
| `light_global_update/linear_2` | (128,512) | local_gate (relu init, gelu) |
| `light_global_update/linear_3` | (128,512) | chain_gate (index_mean over chain) |
| `light_global_update/linear_4` | (128,512) | batch_gate (index_mean over batch) |
| `light_global_update/linear_5` | (512,128) | out proj, zeros, **bias=True** |
| `layer_norm_5` | 128 | prenorm → `local_norm` |
| `linear_14` | (128,60) | position update A·3, zeros, no bias, × `out_scale` |

**`default_ve_scaled`** adds two Linears per pair block (`distance_features(init_pos)`
400→64 and `position_rotation_features(init_pos)` 9→64), shifting all indices:
`linear`(66) `_1`(64) `_2`(3) `_3`(400) `_4`(60) `_5`(9) `_6`(240) `_7`(400 init)
`_8`(9 init) | `_9`…`_17` (2nd block) | `_18`(128,60) position update.

**`multimotif_vp`** inserts `Linear(64)(dmap_mask[...,None])` (w (1,64)) after the
relpos Linear in each pair block, and uses `MotifAdapterDiffusionBlock` (NSB:2099,
still named `"diffusion_block"`) whose first attention runs over
`extract_motif_neighbours(32)`; 21 Linears total.

### 2.2 Sparse IPA — `…/sparse_structure_attn/ada_point_attention/*`

`SparseStructureAttention` (geometric.py:393, `normalize=True`) instantiates
`SparseInvariantPointAttention` (geometric.py:718) because `multi_query=False`.
`heads=8`, `size=32`, `query_points=value_points=8` (hard-coded),
`final_init="zeros"`.

| scope | shape | op |
|---|---|---|
| `layer_norm` | 128 | input LN (`normalize=True`) |
| `qkv` | (128,768) | → q,k,v each (N,8,32); no bias |
| `layer_norm_1` / `layer_norm_2` | 32 / 32 | LN on q / on k |
| `qkv_global/linear` | (128,576) | `LinearToPoints(8*24)` → (N,8,24,3), split 8/8/8 into q_g,k_g,v_g, mapped through residue frames |
| `bias` | (64,8) | pair → per-head attention bias |
| `gamma` | (8,) | `Constant(log(exp(1)-1))`, used as `softplus(gamma)` |
| `project_out` | (1024,128) | concat[out_pair 512, out_scalar 256, out_point 192, out_norm 64] → 128, zeros |

```
w_C = sqrt(2/(9*query_points));  w_L = sqrt(1/3)
dfactor = softplus(gamma)[1,1,H] * w_C / 2
dist    = dfactor * ((q_g[:,None] - k_g[nbr])**2).sum((-1,-2))
dot     = sqrt(1/size) * einsum("ihc,ijhc->ijh", q, k[nbr])
logits  = w_L * (dot + bias - dist)          # -1e9 where ~(mask & nbr!=-1)
attn    = softmax(logits, axis=-2)           # over K, then re-masked to 0
out_point = frames.apply_inverse_to_point(einsum("ijh,ijhpc->ihpc", attn, v_g[nbr]))
```

Frames from `extract_aa_frames` (`utils/geometry.py:51`): Gram-Schmidt
`Rot3Array.from_two_vectors(C-CA, N-CA)`, origin CA.

### 2.3 Remaining scopes (default_vp)

**`SD/diffusion/`**
- `~prepare_features/mlp/linear` (1725,512), `linear_1` (512,128) — inputs =
  time(1+256) + local_pos(60) + `distance_rbf(‖local_pos‖,0,22,**64**)`=1280 +
  LN(prev.local)(128) = 1725. `layer_norm`=LN(prev.local), `layer_norm_1`=LN(MLP out),
  `layer_norm_2`=LN(condition).
- `linear` (128,3) — dssp decoder, zeros, log_softmax (aux).
- `mlp/linear` (529,256), `mlp/linear_1` (256,14) — `get_angle_positions`: inputs =
  local(128)+local_pos(60)+`distance_rbf(…,0,10,16)`(320)+one_hot(aa,21) = 529;
  out 7×2 torsions, L2-normalised → `single_protein_sidechains`.

**`SD/diffusion/aa_decoder/`** (`aa_decoder_kind="adm"`, depth 3)
- `linear` (21,128) glorot — one-hot(aa,21) added to `local`.
- `aa_decoder_stack/__layer_stack_no_state/aa_decoder_block/*` (leading axis 3):
  `linear`(66,64) relpos, `_1`(400,64), `_2`(60,64), `_3`(9,64), `_4`(240,64),
  `_5`(21,64) aa one-hot → `layer_norm`(64) → gather at neighbours →
  `layer_norm_1`(64), `mlp`(64→128→64), `layer_norm_2`(128) prenorm,
  `sparse_structure_attn/ada_point_attention/*`, `layer_norm_3`(128) prenorm,
  `light_global_update/*` = **`EncoderUpdate`** (NSB:2715 — same haiku name but
  only 3 Linears: (128,512),(128,512),(512,128) + mlp 60→256→128).
- `aa_decoder_stack/layer_norm`(128), `layer_norm`(128), `linear_1`(128,20) zeros → logits.
- Neighbours: `extract_neighbours(0, 32, 0)` → 32 spatial.

**`SD/diffusion/distogram_decoder/`** (aux, not needed to sample)
`linear`(66,64), `_1`(400,64), `_2`(60,64), `_3`(9,64), `_4`(240,64),
`_5`(128,64) local[:,None], `_6`(128,64) local[nbr], `layer_norm`(64),
`mlp`(64→128→64 bins), log_softmax.

**`SD/~prepare_condition/condition/`** — `linear`(21,128)+`layer_norm` (aa),
`linear_1`(3,128)+`layer_norm_1` (dssp), `linear_2`(3,128)+`layer_norm_2`
(per-chain dssp mean). All glorot, bias=True. Output = masked sum of the three.

**`SD/~encode/encoder/`** (unused de-novo)
`~prepare_features/vector_linear` (5,15) — `VectorLinear` applied on the *atom*
axis (equivariant vector mixing), then `vector_mean_norm`;
`~prepare_features/mlp` (400,512),(512,128) — dirs(60)+`distance_rbf(…,0,22,**16**)`(320)+log(d+1)(20);
`self_cond_stack/__layer_stack_no_state/encoder_block/*` (leading axis 2) with
`Update(global_update=False)`; `encoder/mlp` (128,256),(256,45) — 15 pseudo-atom
offsets. Encoder neighbours: `get_spatial_neighbours(48)`.

## 3. Resolved configs

### `default_vp` (config lines 3-59, nothing inherited)

```python
augment_size=15, encoder_depth=2, local_size=128, pair_size=64,
num_neighbours=48, resi_dual=False, relative_position_encoding_max=32, factor=4,
heads=8, key_size=32, multi_query=False, diffusion_depth=6, block_size=1,
diffusion_kind="vpfixed", pos_mean_sigma=1.6, pos_std_sigma=1.4,
sigma_data=10.0, preconditioning="vp", diffusion_time_scale="cosine",
diffuse_sequence=False, aa_decoder_kind="adm", aa_decoder_depth=3,
p_clip=1.0, p_drop_dssp=0.2, time_embedding=True,
local_neighbours=16, fape_neighbours=64,
pos_weight=2.0, trajectory_weight=1.0, x_weight=2.0, rotation_weight=2.0,
rotation_trajectory_weight=1.0, violation_weight=0.1, local_weight=10.0,
aa_weight=10.0, fape_weight=1, fape_trajectory_weight=0.5,
min_size=50, max_size=None
```

Absent ⇒ `None` (falsy): `minimal, multi_motif, atom, atom14_ca, atom14_learned,
atom_size, encode_atom14, mask_atom14, aa_trajectory, distogram_trajectory,
distogram_no_pos, nonequivariant_dense, repeat, equivariance, kabsch, linear_aa,
refine, sample_aa, temperature, soft_lddt, use_omap, use_pair, x_late, symm,
sym_noise, block_symmetry, cyclic, condition_time_embedding, pos_min_sigma,
pos_max_sigma, update_init, eval`.

Effective: A = 20 atoms/residue; `out_scale = sigma_data = 10.0`; in/skip/loss = 1.
`config.eval=True` is set by scripts but **never read** in NSB.
`relative_position_encoding_max`, `min_size`, `max_size` are **never read** either
(relpos count is hard-coded 32 at every call site).

### `default_ve_scaled` = `default_vp` + 4 overrides

```python
preconditioning="edm_scaled"   diffusion_kind="edm"
pos_weight=200.0               trajectory_weight=100.0
```
⇒ `EDMDiffusionStack`; `scale = edm_scaling(10.0, 1.0, alpha=1, beta=t)` gives
`in=1/sqrt(100+t²)`, `skip=100/(100+t²)`, `out=10t/sqrt(100+t²)`;
`init_pos = 10·in·pos`, `pos ← skip·pos` before block 1, `init_pos` fed to both
pair-feature calls. Time embedding switches to `concat(log(t)/4, fourier(log(t)/4,256))`.

### Others
- `default_vp_scaled` = `+diffusion_kind="vp"` only ⇒ **same params**; noising uses
  the per-batch CA std instead of fixed `sigma_data`.
- `multimotif_vp` = `+multi_motif=True` ⇒ `MotifAdapterDiffusionBlock` + dmap_mask Linear.

## 4. Haiku constructs with no 1:1 PyTorch mapping

1. **`layer_stack` / `block_stack`** (`basic.py:136`, `aflib/model/layer_stack.py`).
   Module named `__layer_stack_no_state` (Encoder/AADecoder stacks) or
   `__layer_stack_with_state` (diffusion stacks). **Every param inside gets a
   leading axis of `depth`** (6/3/2) — `unbind(0)` into separate modules.
   Implemented with `hk.scan`; RNG threaded by `split` per layer
   (layer_stack.py:133), so per-layer random-neighbour draws differ.
   `with_state=True` also stacks per-layer output → `trajectory (6,N,20,3)`.
   `block_size>1` would nest two layer_stacks and hits broken code at NSB:1634.
2. **`hk.remat`** on every stack body — gradient checkpointing only.
3. **`hk.fori_loop`** NSB:434 (training self-cond) and NSB:1539 (`AADecoder.sample`,
   only with `sample_aa`).
4. **`stop_gradient`** NSB:428/434/1315/1359/1406 — training only.
5. **`jax.vmap`** NSB:1299/1324 over the trajectory axis — training only.
6. **RNG at inference is real**: `get_random_neighbours` calls `hk.next_rng_key()`
   in *every* block (32 random in `extract_neighbours`, 16+16 in
   `extract_condition_neighbours`), plus distogram neighbours, plus
   `AADecoder.train`'s `diffuse_sequence`, plus noising. **Bit-exact JAX↔PyTorch
   matching of a full sample is impossible** without replicating threefry.
7. **No dropout anywhere**; `c.eval` never read.
8. **Initialisers** (`basic.py:106`), haiku `VarianceScaling`,
   `distribution="truncated_normal"` (σ = sqrt(scale/fan)/0.87962566103423978,
   truncated at 2σ):
   - `"linear"` = fan_in, scale 1.0 (Linear default)
   - `"relu"` = fan_in, scale 2.0
   - `"glorot"` = fan_avg, scale 1.0, **uniform** (limit sqrt(3·scale/fan_avg))
   - `"zeros"` = Constant(0.0); bias always Constant(0.0)
   Usage: pair Linears `linear`/no bias; MLP hidden `relu`; `Update` condition→zeros,
   local_update→linear, gates→relu, out→zeros+bias; `update_positions`→zeros;
   IPA qkv/qkv_global/bias→linear, project_out→zeros; AADecoder aa-embed and
   `Condition`→glorot; final logits→zeros.
9. **Unusual params**: `gamma (heads,)` init `Constant(log(exp(1)-1))≈0.5413`;
   `VectorLinear` stores ordinary `w [5,15]` applied on the **atom axis**.
10. **Haiku naming**: a module built in a method other than `__call__` gets
    `parent/~method/name` (hence `~encode/encoder`, `~prepare_condition/condition`,
    `diffusion/~prepare_features/…`). Duplicate names get `_1,_2,…` **in creation
    order**, and the counter is shared across the whole scope — which is why
    `linear_0..linear_14` all sit directly under `diffusion_block`.
11. **Flat layout**: `index_mean`/`index_sum` are scatter-reduces over
    `batch_index`/`chain_index`. SALAD packs a whole batch into one flat residue
    axis with no padding and no batch dim. Keep this layout.

## 5. Dead / experimental on the shipped path

Never-taken branches in `Diffusion.__call__` (NSB:1005-1030): `c.atom`
(`AtomDiffusionBlock`, `AllAtomInteraction`), `c.minimal`, `c.distogram_trajectory`,
`c.multi_motif` (except multimotif_vp), `c.nonequivariant_dense`, `c.repeat`.

Also dead: `Atom14Encoder`; `equivariance=="semi_equivariant"`;
`SparseInvariantMultiQueryAttention` (multi_query False); `SparseAttention`,
`SparseStructureMessage`, `SparsePairUpdate`, `flash_ipa`, `triangle`;
`use_omap`, `cyclic` (unless requested), `block_symmetry`, `symm`/`sym_noise`,
`kabsch`, `soft_lddt`, `x_late`, `mask_atom14`, `linear_aa`, `refine`,
`aa_trajectory`, `sample_aa`/`temperature`, `distogram_no_pos`, `use_pair`,
`atom_size`, `diffuse_sequence`.
`c.condition_time_embedding` is **broken** (passes a function object to a Linear)
— would crash; never enabled.

Training-only, skippable for inference: `StructureDiffusion.__call__`,
`prepare_data`, training `prepare_condition` (NSB:223), `Diffusion.loss`
(NSB:1182-1432, ~250 lines), `get_sigma_edm`, training `apply_diffusion`.

Aux heads not needed to sample: `dssp_decoder`, `DistogramDecoder` — but they do
consume RNG.

**Minimum module set for a de-novo port**: `Condition` → `prepare_features` →
6×`DiffusionBlock` → `AADecoder` (3 blocks) → `get_angle_positions` /
`single_protein_sidechains`. ≈101 of the 132 param groups.
