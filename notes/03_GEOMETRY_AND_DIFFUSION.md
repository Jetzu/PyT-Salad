# SALAD geometry, sparse attention and noise schedules — port notes

## 0. Global data layout (critical)

**SALAD never uses a batch dimension.** Everything is a flat residue axis `N`
plus companion index vectors:

- `residue_index`, `chain_index`, `batch_index`, `mask` — all `(N,)`
- `pos (N, A, 3) float32`, `A = 5 + augment_size` (=20) or 14 for atom14 models.
  **Atom index 1 is always CA, index 4 is CB.**
- Multi-protein batching = concatenation along `N` + `batch_index`. All "global"
  reductions are segment ops keyed on `batch_index` / `chain_index`.
- Neighbour arrays are `(N, K) int32` with **`-1` meaning "no neighbour"**.

## 1. Sparse neighbourhood machinery

### 1.1 The primitive — `get_neighbours(count)` (`utils/geometry.py:242-271`)

```python
distance = where(mask, distance, inf)              # mask is a PAIR mask (N,N)
if neighbours is not None:                          # forbid re-selection
    distance = distance.at[index[:,None], neighbours].set(inf-where)
knn = argsort(distance, -1)[..., :count]            # ascending, NOT topk
knn = where(distance[index[:,None], knn] < inf, knn, -1)
if neighbours is not None:
    knn = concatenate((neighbours, knn), -1)        # (N, K0+count)
```

Semantics to reproduce exactly:
- A **dense `(N,N)` distance matrix is materialised**; selection is `argsort` +
  head-slice, not `topk`. `jnp.argsort` is **stable** → use
  `torch.sort(..., stable=True)`. Ties matter (integer sequence distances,
  all-`inf` rows).
- **Chaining**: each stage appends `count` *new* columns and sets already-chosen
  entries to `inf`. Final `K` is the **sum** of stage counts.
- `-1` is a real gather index in JAX (wraps to last residue). Downstream code
  always masks on `(neighbours != -1)` and never trusts the gathered value.
  Torch advanced indexing `x[idx]` wraps the same way, but `torch.gather` errors
  — clamp to 0 and mask, or keep advanced indexing.

### 1.2 Stage builders (`utils/geometry.py`)

| fn | line | distance | mask |
|---|---|---|---|
| `get_index_neighbours(count)` | 293 | `abs(resi_i - resi_j)` | same_item · same_chain · mask⊗mask |
| `get_spatial_neighbours(count)` | 303 | `‖pos_i − pos_j‖` (pos = CA slice) | mask⊗mask · same_item |
| `get_random_neighbours(count)` | 315 | Gumbel-top-k (below) | same |
| `get_contact_neighbours(count)` | 353 | `U(0,1)` where `pair_condition.any(-1)` else inf | passed in |

**`get_random_neighbours`** (`:317-337`) — Gumbel-top-k sampling **without
replacement with probability ∝ 1/d³**:
```python
weight   = -3 * log(distance + 1e-6)
uniform  = uniform(key, minval=1e-6, maxval=1-1e-6)
gumbel   = log(-log(uniform))          # == -G,  G ~ Gumbel(0,1)
distance = -(weight - gumbel)          # sort ascending on -logits
```
Torch: `logits = -3*log(d+1e-6) + (-log(-log(U)))`, sort `-logits` ascending.

Its first argument is **overloaded**: a square `(N,N)` ndarray is treated as an
already-computed distance matrix; `(N,3)` becomes a `Vec3Array`; a `Vec3Array`
gets a pairwise norm. Both paths are used.

### 1.3 Composite extractors

`extract_neighbours(num_index, num_spatial, num_random)` (`geometry.py:273`):
index stage → spatial stage on `pos[:,1]` (CA) → random stage on CA.

`extract_condition_neighbours(ni, ns, nr, nb)` (NSB:1805): index stage; CA
distance **overridden by `pair_condition["dmap"]` where `dmap_mask`**; spatial;
random; then a block-contact stage with `dist = where(flags.any(-1), 1.0, inf)`.

`extract_motif_neighbours(count)` (NSB:1833): `get_neighbours(count)(dmap, dmap_mask)`.

> **Name-collision trap**: `structure_autoencoder.py:1283` *redefines*
> `extract_neighbours` with a completely different algorithm (sequence-window +
> hard cutoff + one single Gumbel call), shadowing the import for all uses in
> that file. The diffusion model uses the `geometry.py` version. Do not conflate.

### 1.4 Concrete K values

| call site | composition | K |
|---|---|---|
| `DiffusionBlock` current pos (NSB:2188) | `extract_neighbours(16,16,32)` | **64** |
| `DiffusionBlock` self-cond (NSB:2192) | `extract_condition_neighbours(16,16,16,16)` | **64** |
| `MotifAdapterDiffusionBlock` (NSB:2113) | `extract_neighbours(16,16,32)` + `extract_motif_neighbours(32)` | 64 and 32 |
| `Encoder.prepare_features` (NSB:725) | `get_spatial_neighbours(48)` on CB | 48 |
| `AADecoder` (NSB:1510) | `extract_neighbours(0,32,0)` | 32 |
| distogram/FAPE supervision (NSB:1043) | `get_random_neighbours(64)` on CB | 64 |

**Neighbours are recomputed inside every block from the current (already
updated) positions.** `(N,N)` distance + argsort per block per attention — the
single biggest performance item in the port.

### 1.5 Gather/scatter idioms

```python
pair_mask = mask[:,None] * mask[neighbours] * (neighbours != -1)   # (N,K)
x_j       = x[neighbours]                                          # (N,K,...)
pair_ij   = pair[index[:,None], neighbours]                        # (N,K,C)
pair      = pair.at[index[:,None], neighbours].add(update)         # scatter-add
```
Torch: advanced indexing works identically; scatter-add →
`index_put_(..., accumulate=True)`.

### 1.6 Segment ops (`geometry.py:516-645`)

All reduce by index then **broadcast back to input size** (not num-segments):
`index_sum` :516, `index_max` :539, `index_mean` :563 (weighted, 1e-6 floor),
`index_var` :593, `index_std` :612 (`sqrt(var+1e-6)`), `index_count` :630.
Torch: `index_add_` / `scatter_reduce_(reduce='amax')`, then `result[index]`.

> ⚠️ `index_mean` does `data *= weight` and `index_kabsch` does `x -= x_center`
> — pure in JAX, **destructive in torch**. `clone()` first.

Others used: `index_kabsch` :647 (SVD on `stop_gradient(cov)`, det-sign flip),
`index_align` :677, `apply_alignment` :703, `unique_chain` :727 (`lax.scan` →
replace with `cumsum` of boundary flags), `positions_to_ncacocb` :751,
`compute_pseudo_cb` :488 (constants `[-0.58273431, 0.56802827, -0.54067466]`),
`assign_sse` :769 (P-SEA, two scans), `interactions` :833.

## 2. Attention

### 2.1 `SparseInvariantPointAttention` (`geometric.py:718`) — the workhorse

See `02_ARCHITECTURE.md` §2.2 for shapes and math.
**Fully SE(3)-invariant output**: `q_g/k_g/v_g` are produced in local frames and
pushed to global via `apply_to_point`, so `|q_g_i − k_g_j|` is invariant; `v_g`
is aggregated globally then pulled back with `apply_inverse_to_point`.
Reflections are not handled (frames are proper rotations by construction).
Called with `pos / c.sigma_data` — i.e. **positions in units of 10 Å**
(NSB:2211, 2221).

`LinearToPoints` (`geometric.py:264`): `Linear(size*3, no bias)` → reshape
`(...,size,3)` → `frames[...,None].apply_to_point`.

### 2.2 Other variants (all dead for shipped configs)

- `SparseAttention` :504 — no point terms. Used only by structure_autoencoder.
- `SparseSemiEquivariantPointAttention` :636 — translation-equivariant only.
  Has a latent bug at :664: `qkv_g` is computed then immediately overwritten.
- `SparseInvariantMultiQueryAttention` :439 — **dead**; `multi_query=False`
  everywhere and the einsum would not broadcast (h=H vs h=1).
- `DenseNonEquivariantPointAttention` :559 — experimental.
- `FlashPointAttention` (`flash_ipa.py:14`) — dense IPA refactored into one
  standard SDPA call. ⚠️ `overscale_dfactor=True` (the default) does **not**
  reproduce standard IPA: it applies `w_C` twice, giving
  `softplus(γ)·w_C²/4` instead of `softplus(γ)·w_C/2`. `use_pair=True` is broken.
- **Triangle ops (`geometric/triangle.py`) do not run**: `jnp.swapaxis` does not
  exist and the reshape is malformed; no call site. **SALAD's released models
  contain no triangle multiplicative update — do not port it.**

### 2.3 Pair-feature builders

Production one is `diffusion_pair_features(c)` (NSB:2034): sum of zero-bias
`Linear(pair_size)` projections of relpos one-hot(66), motif dmap RBF(64),
optional omap(9), pair flags(3), `distance_features`(400),
`direction_features`(60), `position_rotation_features`(9),
`pair_vector_features`(240), plus the same distance+rotation on `init_pos` when
present; then LayerNorm + a 2-layer gelu MLP.

Component functions (`geometric.py`): `distance_features` :23 (5×5 atom pairs ×
16 RBF), `direction_features` :45, `rotation_features` :114 /
`position_rotation_features` :131, `pair_vector_features` :164.
All other builders in that file are unused by shipped models.

`sequence_relative_position(32, one_hot, cyclic, …)` (`geometry.py:181`):
`dist = resi_i − resi_j` clipped to ±32, +32; cross-chain/cross-batch → bin 65;
`count_total = 66`. Cyclic wrapping uses `index_count` lengths and `dist % length`.
(`identify_ends=True` has an off-by-two bug; unused.)

RBF helpers (`geometry.py:369-470`): `gaussian_rbf(x,c,step) = exp(-(x-c)²/step²)`
— **note `step²`, not `2σ²`**; `distance_rbf(d,0,22,64)` with
`step=(max-min)/bins`, `centers = min + arange(bins)*step + step/2`.
`spherical_harmonics` :472 needs `lpmn_values` (no torch equivalent) but is unused.

## 3. Vec3Array / Rot3Array / Rigid3Array

`aflib/model/geometry/struct_of_array.py:181` is a decorator turning a frozen
dataclass into a struct-of-arrays container **and** a registered JAX pytree. It
injects `shape`, `__len__`, `__getitem__` (same index applied to every field),
`replace`, `dtype`. The SoA layout exists to avoid tiny matmuls and bf16
contamination on TPU (`vector.py:35-44`).

Conversions: `Vec3Array.to_array()` → `stack([x,y,z],-1)`;
`Rot3Array.to_array()` → `(...,3,3)` **row-major**;
`Rigid3Array.to_array()` → `concat([rot, trans[...,None]], -1)` = `(...,3,4)`;
`from_array(a)`: `rot=a[...,:3]`, `trans=a[...,-1]`.

Methods actually used in `salad/modules` (grep counts): `Vec3Array.from_array`
309, `.to_array()` 201, `.norm()` 98, `.apply_inverse_to_point` 47,
`.normalized()` 35, `.rotation` 33, `.apply_to_point` 22, `.norm2()` 17,
`.inverse()` 16, `.cross` 13, `.translation` 10,
`Rot3Array.from_two_vectors` 6, `.dot` 5, `Rigid3Array.from_array` 7, plus
arithmetic.

### Recommended PyTorch representation

**Plain tensors with a trailing dim of 3 / (3,3) / (3,4).** SoA buys nothing in
torch and the pytree machinery has no analogue.

- `Vec3Array` → `Tensor[...,3]`
  - `norm()` → `sqrt(clamp((v*v).sum(-1), min=1e-12))` — **clamp before sqrt**;
    `clamp(sqrt(x), min=1e-6)` gives NaN gradients at 0.
  - `norm2()` → `(v*v).sum(-1)` (no clamp)
- `Rot3Array` → `Tensor[...,3,3]` row-major
  - `apply_to_point(p)` → `einsum('...ij,...j->...i', R, p)`
  - `apply_inverse_to_point(p)` → `einsum('...ji,...j->...i', R, p)`
  - `from_two_vectors` → explicit Gram-Schmidt (`rotation_matrix.py:80-101`):
    `e0 = normalize(v0)`, `e1 = normalize(v1 − (v1·e0)e0)`, `e2 = e0×e1`, columns `(e0,e1,e2)`
- `Rigid3Array` → `(R [...,3,3], t [...,3])` tuple or packed `[...,3,4]`
  - `apply_to_point(p) = Rp + t`; `apply_inverse_to_point(p) = Rᵀ(p − t)`

The one friction point is the broadcast idiom
`frames[:,None,None].apply_inverse_to_point(points)`. **Write a ~30-line `Rigid3`
helper whose `__getitem__` unsqueezes before the trailing dims** — it removes all
the porting risk.

Frame construction to reproduce exactly (`geometry.py:19-62`):
```
make_transform_from_reference(a=N, b=CA, c=C):
    rotation    = Rot3Array.from_two_vectors(C - CA, N - CA)
    translation = CA
extract_aa_frames(pos): local = rigids[..., None].apply_inverse_to_point(pos)
```

## 4. Noise schedules

### 4.1 σ(t)

Training-time (NSB:62-95), selected by `c.diffusion_time_scale`:

| name | formula |
|---|---|
| `sigma_scale_cosine(t, s=0.01)` | `ᾱ = cos((t+s)/(1+s)·π/2)²/cos(s/(1+s)·π/2)²`, clip, `σ=sqrt(1-ᾱ)`. σ(0)=0, σ(1)=1 |
| `sigma_scale_framediff(t, bmin=.1, bmax=20)` | `Gs = t·bmin + ½t²(bmax−bmin)`, `σ=sqrt(1−e^{−Gs})` |
| `sigma_scale_none/linear` | `σ = t` |

Inference-only extras (`inference/utils/options.py`):
- `sigma_scale_ve(t, sigma_max=80, rho=7)` :79 — Karras/EDM,
  `σ = (σmax^{1/ρ} + (1−t)(σmin^{1/ρ} − σmax^{1/ρ}))^ρ`, σmin=0.05
- `sigma_scale_edm(t)` :85 — `exp(1.6 + 1.4·Φ⁻¹(clip(t,.001,.999)))`; needs
  `torch.special.ndtri`
- `cloud_std_default(n)` :109 — `n**0.4 + U(0,1)*3`

> `timescale_pos` is an **inference-only CLI argument, not a config key**. It maps
> `raw_t = 1 − step/num_steps` to the σ written into `data["t_pos"]`.

### 4.2 Forward noising (`utils/diffusion.py`)

| fn | line | formula |
|---|---|---|
| `diffuse_coordinates_edm` | 27 | **VE**: `x + t·ε`, CA-noise centroid removed per batch item |
| `diffuse_atom_cloud` | 69 | **VP**: `x·sqrt(1−t²) + t·ε`, `ε ~ N(0, cloud_std²)`, centroid removed. `flow=True` → `x·(1−t) + t·ε` |
| `position_std` | 52 | per-segment CA std, **averaged over the 3 axes then sqrt** (≈ Rg/√3) |
| `diffuse_sequence` | 12 | Bernoulli(p=t) masking to token 20 |
| `fourier_time_embedding(t, size=128, base=10000)` | 176 | `exponent = 2·arange(size//2)/size`, `concat(sin, cos)` |

Training σ sampling (NSB:663): log-normal `σ = exp(1.6 + 1.4·z)` (mean ≈13 Å,
heavy tail), one sample **per batch item**. VP/flow sample `t ~ U(0,1)` per item,
then `t_pos := SIGMA_SCALE[time_scale](t)`.

### 4.3 Preconditioning (NSB:940-987)

```python
def edm_scaling(sd, sn, alpha=1, beta=1):
    denom = alpha²sd² + beta²sn²
    in_    = 1/sqrt(denom);  out = beta·sd·sn/sqrt(denom)
    skip   = alpha·sd²/denom; loss = 1/out²

preconditioning_scale_factors(c, t, sd):
    "edm"/"edm_scaled" -> edm_scaling(sd, 1.0, alpha=1, beta=t)
    "flow"             -> {in:1, out:sd, skip:1, loss:1/max(t²,1e-6)}
    "vp"               -> {in:1, out:sd, skip:1, loss:1}
```

Where the scales land: `preconditioning=="edm"` scales `pos *= skip` before the
stack (NSB:1055); `"edm_scaled"` uses `EDMDiffusionStack` (NSB:1697) which sets
`init_pos = sd·in·pos` and `pos = skip·pos`; every block's position update uses
`out_scale = …["out"]` — for VP a **constant 10.0 Å**, for VE a per-residue
`t·sd/sqrt(sd²+t²)`.

Time embedding (NSB:1147): `vp_time_embedding = concat(t, fourier(t,256))` → 257
dims for `diffusion_kind ∈ {vp, vpfixed, chroma, flow}`; otherwise
`edm_time_embedding = concat(log(x)/4, fourier(log(x)/4, 256))`.
(`"flow_scaled"` is missing from that tuple — a latent bug; unused.)

### 4.4 Config → schedule

| config | diffusion_kind | preconditioning | time_scale | cloud_std |
|---|---|---|---|---|
| `default_vp` | `vpfixed` | `vp` | `cosine` | fixed `sigma_data=10` |
| `default_vp_scaled` | `vp` | `vp` | `cosine` | `position_std(x)` / `--cloud_std` |
| `default_ve` | `edm` | `edm` | — | — |
| `default_ve_scaled` | `edm` | `edm_scaled` | — | — |

README-recommended inference settings:
`default_vp` → `--timescale_pos "cosine(t)" --cloud_std none`;
`default_vp_scaled` → `+ --cloud_std "default(num_aa)"`;
`default_ve_scaled` → `--timescale_pos "ve(t)" --prev_threshold 0.95..0.99`.

## 5. JAX primitives needing attention

- **Scatter** (`.at[].add/.max/.set`) → `index_add_`, `scatter_reduce_`,
  `index_put_(accumulate=True)`.
- **`lax.scan` / `fori_loop`** — `unique_chain` and block indexing collapse to
  `cumsum` of boundary flags; `assign_sse` needs a real sequential loop but is
  **data preprocessing** — do it in numpy on CPU.
- **`layer_stack`** — see `02_ARCHITECTURE.md` §4.1. Gradient checkpointing →
  `torch.utils.checkpoint`.
- **`jax.vmap`** — always over a leading trajectory axis; flatten into batch or
  rely on broadcasting. **No vmap over ragged data anywhere.**
- **`jax.nn.dot_product_attention`** (flash_ipa only) → `F.scaled_dot_product_attention`
  with `scale=1.0`. The JAX call needs equal key/value head-dim, worked around by
  zero-padding — SDPA has no such restriction, drop the pad.
- **`take_along_axis`** — not used anywhere. `jnp.take(..., mode='clip')` appears
  in `aflib/model/utils.py:75` (index clamping, matters for `torsion_angles_to_frames`).
- **`custom_jvp`/`custom_vjp`** — none in the repo.
- **float64** — never enabled. Precision is handled by explicit float32 casts at
  the geometry boundary. **Port rule: do all frame construction, point projection
  and norms in fp32 even under AMP.**

### Numerically sensitive spots

- `Vec3Array.norm`: `sqrt(max(dot, 1e-12))` — clamp **before** sqrt.
- `lax.rsqrt` in `VectorLayerNorm`/`vector_std_norm`, each with a `max(…,1e-6)` guard.
- `gaussian_rbf` uses `step²`, not `2σ²`.
- `index_mean` divides by `max(count, 1e-6)`; `index_std` adds 1e-6 inside sqrt.
- Attention masks with literal `-1e9`, then a **second** `where(mask, attn, 0)`
  after softmax. **Keep both** — with fully-masked rows the post-softmax zeroing
  is what prevents garbage.
- `argsort` stability (see §1.1).
- `index_kabsch` SVD det-sign flip produces `...ba` — the **transpose** of the
  naive product. Keep it.
