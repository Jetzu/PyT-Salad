# PyTorch port specification: noise schedules, forward noising, sampling loop, PDB output

Reference: `/home/hpc/v181bb/v181bb10/SALAD_torch/upstream/salad` (JAX, read-only).
Target: `/home/hpc/v181bb/v181bb10/SALAD_torch/salad_torch`.
Existing notes already agreed: `notes/01_INFERENCE_PATH.md`, `notes/03_GEOMETRY_AND_DIFFUSION.md`.

Conventions already established in `salad_torch` that you must follow:
* No batch dimension. Flat residue axis `N`, companion `residue_index / chain_index / batch_index / mask`, all `(N,)`.
* `pos` is `(N, A, 3) float32`, `A = 5 + augment_size = 20` for `default_vp` (`A = 14` only if `encode_atom14`). **Atom 1 is CA, atom 4 is CB.**
* Segment ops live in `salad_torch/geometry/segment.py` with signature `index_mean(data, index, mask, weight=None, apply_mask=True)`; `mask` is expected to already broadcast against `data` (callers pass `mask[..., None]`).
* Preconditioning already exists at `salad_torch/diffusion/precond.py` (`edm_scaling`, `preconditioning_scale_factors`). Do not duplicate.
* `salad_torch/diffusion/__init__.py` and `salad_torch/io/__init__.py` are currently empty — this spec fills them.

Suggested new files:
* `salad_torch/diffusion/schedules.py` — §1
* `salad_torch/diffusion/forward.py` — §2
* `salad_torch/diffusion/sampling.py` — §3
* `salad_torch/io/residue_constants.py`, `salad_torch/io/atom_layout.py`, `salad_torch/io/pdb.py` — §4

---

# 1. NOISE SCHEDULES

## 1.1 Exact formulas

### `sigma_scale_cosine(t, s=0.01)` — `salad/modules/noise_schedule_benchmark.py:62-72`

```
alpha_bar = cos((t + s) / (1 + s) * pi / 2) ** 2
alpha_bar = alpha_bar / cos(s / (1 + s) * pi / 2) ** 2
alpha_bar = clip(alpha_bar, 0.0, 1.0)
sigma     = sqrt(1 - alpha_bar)
```
Constants: `s = 0.01` (default, never overridden anywhere in the repo). Clipping is `[0, 1]` on `alpha_bar`, **not** on sigma.
Reference values (float32): `t=1.0 -> 1.0`; `t=0.8 -> 0.95233`; `t=0.5 -> 0.72925`; `t=0.2 -> 0.320385`; `t=0.0 -> 0.0`.

### `sigma_scale_framediff(t, bmin=0.1, bmax=20.0)` — `noise_schedule_benchmark.py:75-79`

```
Gs    = t * bmin + 0.5 * t**2 * (bmax - bmin)
sigma = sqrt(1 - exp(-Gs))
```
No clipping. `t=1 -> Gs=10.05 -> sigma≈0.999997`; `t=0 -> 0`.

### `sigma_scale_none` / `sigma_scale_linear` — `noise_schedule_benchmark.py:82-84`

```
sigma = t
```
`SIGMA_SCALE` registry (`noise_schedule_benchmark.py:87-92`) maps `{"cosine", "framediff", "none", "linear"}`, with `none` and `linear` bound to the *same* function.

### `sigma_scale_ve(t, sigma_max=80.0, rho=7.0)` — `salad/inference/utils/options.py:79-83` (duplicated verbatim at `eval_noise_schedule_benchmark.py:72-75`)

```
sigma_min = 0.05                      # hard-coded local, not a parameter
sigma = (sigma_max**(1/rho) + (1 - t) * (sigma_min**(1/rho) - sigma_max**(1/rho))) ** rho
```
This is the Karras/EDM rho-schedule. Increasing in `t`. `t=1 -> 80.0`; `t=0 -> 0.05`; `t=0.2 -> 0.4618`; `t=0.8 -> 33.66`.
**Note the signature quirk:** README:605 documents `ve(t, sigma_max=80.0)` as a keyword call, so the restricted parser must support keyword arguments (see §1.3). README:530 also mentions `sigma_max = 100` was used for some manuscript results.

### `sigma_scale_edm(t)` — `options.py:85-90` (duplicate at `eval_...:77-79`)

```
pos_mean_sigma = 1.6
pos_std_sigma  = 1.4
sigma = exp(pos_mean_sigma + pos_std_sigma * Phi_inv(clip(t, 0.001, 0.999)))
```
`Phi_inv` is `jax.scipy.stats.norm.ppf` = the standard-normal inverse CDF = `scipy.special.ndtri`.
**Torch:** `torch.special.ndtri(x)`. The JAX code applies `.astype(jnp.float32)` to the ppf output *before* the affine+exp — i.e. the cast is inside, on the ppf result only.
`t=0.5 -> 4.9530`; `t=1 -> clip to 0.999 -> ndtri=3.090232 -> 374.6`; `t=0 -> clip to 0.001 -> -3.090232 -> 0.06549`.

### `cloud_std_default(num_aa)` — `options.py:109-112` (duplicate `eval_...:92-94`)

```
minval = num_aa ** 0.4
return minval + np.random.rand() * 3
```
* Uses the **numpy global RNG**, not a jax key. It is drawn **once per design**, at `eval_...:557` (`cloud_std = cloud_std_func(num_aa)`), before the step loop.
* `num_aa=100 -> 6.3096 + U(0,3)`.
* Torch: take a `torch.Generator` (or a `numpy.random.Generator`) argument so it is injectable; default to a module-level generator. Document that exact parity with JAX runs is impossible here because it is numpy's Mersenne Twister — but parity *is* achievable by passing the drawn scalar in explicitly.

### Floating-point precision — reproduce this or you will see ~1e-7 drift

JAX runs with x64 disabled. Python-float arithmetic inside these functions stays float64 until it hits the first `jnp.*` call, then collapses to float32:

| function | float64 part | float32 from |
|---|---|---|
| `cosine` | `(t+s)/(1+s)*pi/2` | `jnp.cos(...)` onward |
| `framediff` | `Gs` | `jnp.exp(-Gs)` onward |
| `ve` | **entire function** (pure Python `**`) | only at the later `scaled_t * ones(N, f32)` |
| `edm` | `clip` arg | `ndtri` output cast to f32, then exp in f32 |

Recommended torch implementation: accept `t: float | Tensor`; compute the pre-`cos`/pre-`exp` argument in Python/float64 when `t` is a scalar, then wrap with `torch.tensor(arg, dtype=torch.float32)` for the transcendental. For `ve`, do the whole thing in Python floats and only cast at the broadcast. For tensor `t`, just do everything in the tensor's dtype (float32) — the difference is below float32 eps for realistic `t`.

## 1.2 Torch special functions needed

* `torch.special.ndtri` — for `edm`. Only non-trivial one.
* Everything else: `torch.cos`, `torch.exp`, `torch.sqrt`, `torch.clamp`, `**`.
* `torch.pi` (or `math.pi`) for the cosine schedule; JAX uses `jnp.pi` which is `numpy.pi` (a Python float).

## 1.3 Safe replacement for `parse_timescale` (no `eval`)

Upstream (`options.py:92-107`, identical at `eval_...:82-90`):
```python
def parse_timescale(data):
    data = data.strip()
    predefined = dict(cosine=..., framediff=..., ve=..., edm=...)
    return lambda t: eval(data, None, dict(t=t, np=np, **predefined))
```
Note `parse_cloud_std` (`options.py:114-127`) is the same shape but with `num_aa` and `default` in scope, plus a `"none"` short-circuit **before** any eval.

Also note: in the eval script the string is re-parsed and re-`eval`'d **every step** (`eval_...:597, 600`). Semantically irrelevant (the schedules are pure), so in torch parse once and cache the callable.

### Required expression forms (this is the complete documented+shipped surface)

| form | where it comes from | must support |
|---|---|---|
| `cosine(t)` | default, README:358/399/683/696/853 | named call, 1 positional arg |
| `ve(t)` | README:379/430/729 | same |
| `ve(t, sigma_max=80.0)` | README:605 | **keyword arguments** |
| `edm(t)`, `framediff(t)` | in `predefined` | same |
| `1` | eval script default `timescale_seq="1"` (`eval_...:458`) | bare integer/float literal |
| `t`, `0.5 * t`, `t ** 2`, `1 - t` | README:522-524 "any expression depending on t … raw python + numpy" | binary ops, unary minus, numeric literals |
| `np.sqrt(t)`, `np.exp(-t)`, `np.pi` | README:524 "(as np)" | dotted `np.<name>` attribute + call, and `np.pi`-style constants |
| `none` | `--cloud_std none` only | literal string check before parsing |
| `default(num_aa)` | README:540 | named call in the cloud_std namespace |
| `num_aa ** 0.4` | README:538 | power with float exponent |

### Design: registry + restricted AST evaluator

```python
# salad_torch/diffusion/schedules.py

SCHEDULE_REGISTRY = {           # sigma(t)
    "cosine":    sigma_scale_cosine,
    "framediff": sigma_scale_framediff,
    "none":      sigma_scale_none,
    "linear":    sigma_scale_none,
    "ve":        sigma_scale_ve,
    "edm":       sigma_scale_edm,
}
CLOUD_STD_REGISTRY = {"default": cloud_std_default}

# whitelisted "np.*" names mapped onto torch/math equivalents
NP_NAMESPACE = {
    "pi": math.pi, "e": math.e,
    "exp": ..., "log": ..., "log10": ..., "sqrt": ..., "abs": ...,
    "sin": ..., "cos": ..., "tan": ..., "tanh": ...,
    "clip": ..., "minimum": ..., "maximum": ..., "power": ...,
    "square": ..., "sign": ..., "floor": ..., "ceil": ...,
}
```
Implement `_safe_eval(expr: str, names: dict) -> float | Tensor` with `ast.parse(expr, mode="eval")` and a visitor that accepts **only**:

* `ast.Expression`
* `ast.Constant` where `type(value) in (int, float)` (reject str/bytes/bool/complex/None)
* `ast.Name` whose `id` is a key of `names` (`t` or `num_aa`, plus the registry keys, plus `np`)
* `ast.Attribute` **only** when `node.value` is `ast.Name(id="np")` and `node.attr` is in `NP_NAMESPACE` (reject dunders by construction since the whitelist is explicit)
* `ast.Call` where `node.func` resolves via the two rules above, `node.starargs`/`**kwargs` (`ast.Starred`, `keyword(arg=None)`) are rejected, and every keyword name is a plain identifier
* `ast.BinOp` with `Add, Sub, Mult, Div, FloorDiv, Mod, Pow`
* `ast.UnaryOp` with `UAdd, USub`
* optionally `ast.Tuple` — **not needed**, reject it

Reject everything else with a `ValueError` naming the offending node type. Add a depth cap (e.g. 32) and a length cap (e.g. 512 chars) as cheap DoS guards. Do **not** accept `ast.Subscript`, `ast.Lambda`, comprehensions, `ast.IfExp`, `ast.Compare`, `ast.BoolOp`, `ast.NamedExpr`.

Public API:
```python
def parse_timescale(spec: str | Callable[[float], float]) -> Callable[[float], float]
def parse_cloud_std(spec: str) -> Callable[[int], float | None]   # "none" -> lambda n: None
```
`parse_timescale` must also accept an already-callable argument and return it unchanged (mirrors `Sampler._get_timescale`, `sampling.py:26-30`).

Fast path: if `spec.strip()` matches `^([a-z_]+)\(t\)$` and the name is in the registry, return the registry function directly (skips the AST entirely for the 95% case).

### `parse_out_steps` (`options.py:72-77`, `eval_...:66-69`)
```
"all" -> range(1000)
else  -> [int(c) for c in data.split(",")]
```
Port as `list[int]`, with `"all"` -> `list(range(1000))`. **Do not sort** — the order matters, see §3.4.

### `parse_num_aa` (`options.py:28-70`, `eval_...:40-64`)
Also needed by the sampler driver. `"100:200:c8"`:
```
for idx, size_str in enumerate(spec.split(":")):
    cyclic = size_str.startswith("c");  size = int(size_str.lstrip("c") if cyclic else size_str)
    last_resi = resi[-1] + 50 if resi else 0
    resi  += [last_resi + i for i in range(size)]
    chain += [idx] * size
    cyclic_mask += [cyclic] * size
```
Returns `(num_aa, resi int64/(int32), chain, is_cyclic: bool, cyclic_mask: bool)`.
**Note the +50 rule is `previous residue's index + 50`, so the second chain starts at `n0 - 1 + 50`, not `n0 + 50`.** For `100:100` the indices are `0..99` then `149..248`.

---

# 2. FORWARD NOISING (`salad/modules/utils/diffusion.py`)

## 2.1 `position_std(x, index, mask)` — `diffusion.py:52-67`

```python
ca = x[:, 1]                                       # (N, 3); x is a plain array (N, A, 3)
center = index_mean(ca, index, mask[..., None])    # (N, 3)
ca -= center
return sqrt( index_mean(ca**2, index, mask[...,None]).mean(axis=1)
           - index_mean(ca,     index, mask[...,None]).mean(axis=1) ** 2 )
```
* Output shape `(N,)` float32 — per-residue, but constant within each segment.
* `.mean(axis=1)` averages the three Cartesian axes **before** the `sqrt`, so this is `sqrt(mean_xyz(var))` ≈ `Rg/sqrt(3)`, not the norm-based std.
* The second term is the square of the mean of the already-centred CA, which is ~0 but not exactly 0.
* **Latent bug / numerical hazard:** there is no epsilon and no `clamp(min=0)` before `sqrt`. For a single-residue segment, or for a fully-masked segment, the argument can be a tiny negative float -> `nan`. Reproduce faithfully but I recommend the writer add an opt-in `eps` kwarg defaulting to `0.0` (bit-identical) so it can be raised when debugging.
* **Torch aliasing trap:** `ca = x[:, 1]` is a *view*; `ca -= center` in JAX rebinds, in torch it would mutate the caller's `x`. Use `ca = x[:, 1] - center` (never `-=`).
* `mask` is `(N,)` bool; `mask[..., None]` is `(N,1)`, broadcast against `(N,3)` data — matches the existing `segment.index_mean` contract.

## 2.2 `diffuse_coordinates_edm(key, x, batch_index, t, symm=None)` — `diffusion.py:27-50` (VE)

```
target = normal(key, [N, A, 3])                    # ONE draw, standard normal
center = segment_sum(target[:, 1], batch_index) / max(segment_count, 1e-6)   # (N,3) broadcast back
target = target - center[:, None, :]               # subtract per-batch CA-noise centroid from ALL atoms
if symm: target = symm(target)
result = x + t * target
```
* `t` is passed as `t_pos[:, None]`, shape `(N, 1)`, float32 (`noise_schedule_benchmark.py:567-569`). Because `Vec3Array` stores each component as `(N, A)`, `(N,1)` broadcasts over the atom axis. **In torch with `(N, A, 3)` tensors `t` must be reshaped to `(N, 1, 1)`.**
* Mask is *not* used; the centroid divisor is the raw per-batch residue count, not the masked count. (Contrast with `diffuse_atom_cloud`, which also ignores `mask` for centring — `mask` there is only consumed by `position_std`.)
* The centre is computed from **CA only** (`target[:, 1]`) and subtracted from **all A atoms**.
* The JAX `.at[batch_index].add(...)` pattern produces an `(N, 3)` accumulator indexed by batch id then re-gathered with `center[batch_index]`; exactly `index_sum` semantics. Use `segment.index_sum` with an all-True mask, or equivalently `index_mean(target[:, 1], batch_index, ones_bool)`.

## 2.3 `diffuse_atom_cloud(...)` — `diffusion.py:69-114` (VP / VP-scaled / flow)

Signature: `(key, x, mask, batch_index, t, cloud_std=None, flow=False, residue_std=4.0, correlated_residues=False, symm=None)`.

```
if cloud_std is None:
    x_sigma   = position_std(x.to_array(), batch_index, mask)   # (N,)
    cloud_std = x_sigma[:, None, None]                          # (N,1,1)

if correlated_residues:
    cloud_key, residue_key = split(key, 2)
    target  = cloud_std   * normal(cloud_key,   [N, 1, 3])      # ONE centre per residue, broadcast over A
    target += residue_std * normal(residue_key, [N, A, 3])      # per-atom jitter, std 4.0 A
else:
    target = cloud_std * normal(key, [N, A, 3])

center = segment_mean_over_batch(target[:, 1])                  # CA of the NOISE, per batch item
target = target - center[:, None, :]
if symm: target = symm(target)

if flow: result = x * (1 - t)          + target * t             # linear / flow matching
else:    result = x * sqrt(1 - t**2)   + target * t             # VP
```

Key facts:
* `cloud_std` is one of: a Python float `10.0` (`c.sigma_data`, used for `diffusion_kind in ("vpfixed","flow")`, `noise_schedule_benchmark.py:574-575`), a user scalar from `data["cloud_std"]` (`:576-577`, overrides the fixed value), or `None` -> per-batch `position_std`.
* Broadcasting: with the `(N,1,1)` form, `cloud_std` scales all atoms of a residue by that residue's segment std.
* `correlated_residues` is **never True** in any shipped config or call site (grep: only the default `False`). Implement it for completeness, and note `residue_std=4.0` is applied *unscaled* by `cloud_std`. Also note the correlated branch is the only place where the function consumes **two** RNG draws.
* `flow` is selected by `c.diffusion_kind == "flow"` at inference (`:580`) but by `c.diffusion_kind in ("flow", "flow_scaled")` during training (`:401`) — a latent inconsistency; `flow_scaled` gets VP noising at inference. Irrelevant for shipped configs (none use flow) but reproduce the inference form.
* Line 109 `target = target` is a no-op left in the source.
* The VP mixing coefficient `sqrt(1 - t**2)` with `t = sigma` means the pair `(alpha, sigma) = (sqrt(1-sigma^2), sigma)` — consistent with the cosine schedule's `alpha_bar = 1 - sigma^2`.
* `mask` is used **only** by `position_std`. Masked residues still get noise.

## 2.4 `diffuse_sequence(key, x, t, mask_index=20)` — `diffusion.py:12-25`

```
corrupt_aa = bernoulli(key, p=t, shape=x.shape)    # x.shape == (N,)
out        = where(corrupt_aa, mask_index, x)
return out, corrupt_aa
```
* `t` is `t_seq`, shape `(N,)` float32 -> per-residue masking probability.
* `mask_index = 20` is the MASK token (amino acids are 0..19).
* Torch: `corrupt = torch.rand(x.shape, generator=g, device=..., dtype=torch.float32) < t`. **JAX's `bernoulli` is `uniform(key, shape) < p`, i.e. strictly less-than.** Match that.
* Returns `(out int32/int64, corrupt bool)`.
* At inference with `default_vp`, `t_seq = 1.0` for every step, so every position is masked to 20 — but the draw still happens and still advances the stream.
* **Second call site:** `AADecoder.train` (`noise_schedule_benchmark.py:1556-1559`) draws `t = uniform(key, batch.shape)[batch]` then calls `diffuse_sequence` again. This runs at inference too (`:1079-1080` calls `aa_decoder.train`), so it consumes 2 more keys per model call.

## 2.5 `fourier_time_embedding(t, size=128, base=10_000, scale=1.0)` — `diffusion.py:176-188`

```
exponent    = 2 * arange(0, size // 2, dtype=t.dtype) / size     # (size//2,)
denominator = base ** exponent
result      = (scale * t[..., None]) / denominator               # (..., size//2)
return concat([sin(result), cos(result)], axis=-1)               # (..., 2*(size//2))
```
* Output width is `2 * (size // 2)`, i.e. `size` when even. Called with `size=256` -> 256 dims.
* `arange` takes `t.dtype` (float32), so `base ** exponent` is evaluated in float32. Torch: `torch.arange(size // 2, dtype=t.dtype, device=t.device)`.
* sin block first, cos block second.
* Used as `vp_time_embedding(x) = cat([x[:, None], fourier_time_embedding(x, size=256)], -1)` -> `(N, 257)` (`noise_schedule_benchmark.py:1151-1152`), and `edm_time_embedding(x) = cat([log(x)[:, None]/4, fourier_time_embedding(log(x)/4, size=256)], -1)` (`:1148-1150`). VP form is selected for `diffusion_kind in ("vp","vpfixed","chroma","flow")` (`:1153-1156`); `"flow_scaled"` is missing from that tuple — a latent bug, unused.
* Note the argument is `t_pos`, which is already **sigma**, not raw diffusion time.

## 2.6 Injectable-RNG torch API

Split every noiser into a *pure* transform plus a *sampler*, so a JAX-drawn noise tensor can be dropped in for parity testing.

```python
# salad_torch/diffusion/forward.py

def center_noise_on_ca(target: Tensor, batch_index: Tensor) -> Tensor:
    """target (N,A,3) -> target minus its per-batch CA centroid. No mask."""

def apply_atom_cloud_noise(x, target, t, *, flow=False, symm=None,
                           batch_index) -> Tensor:
    """Pure. x (N,A,3), target (N,A,3) already scaled by cloud_std,
       t (N,1,1). Does the centring, symm, then the VP/flow mix."""

def sample_atom_cloud_noise(shape, cloud_std, *, residue_std=4.0,
                            correlated_residues=False, generator=None,
                            device=None, dtype=torch.float32
                           ) -> Tensor:
    """Draws standard normals and applies cloud_std/residue_std.
       Returns (N,A,3) BEFORE centring."""

def diffuse_atom_cloud(x, mask, batch_index, t, cloud_std=None, flow=False,
                       residue_std=4.0, correlated_residues=False, symm=None,
                       *, noise=None, generator=None) -> Tensor:
    """Convenience wrapper. If `noise` is given it is used verbatim in place of
       `sample_atom_cloud_noise(...)` (i.e. it must ALREADY include the
       cloud_std / residue_std scaling) and no RNG is consumed."""

def diffuse_coordinates_edm(x, batch_index, t, symm=None,
                            *, noise=None, generator=None) -> Tensor
def diffuse_sequence(x, t, mask_index=20,
                     *, uniform=None, generator=None) -> tuple[Tensor, Tensor]
```
For `diffuse_sequence`, inject the **raw uniform** rather than the boolean, so `< t` stays inside the port and you test the comparison too. Add a lower-level `noise_std_normal=` escape hatch on `diffuse_atom_cloud` that takes the *unscaled* `(N,A,3)` (and, for the correlated path, a `(cloud (N,1,3), residue (N,A,3))` pair) — that is the level at which JAX's `jax.random.normal` output can be dumped and replayed.

Return both the result and the realised noise when a debug flag is set (`return_noise=True`), so the noise can be captured from a torch run too.

## 2.7 RNG draw order (exact)

Per model call, `StructureDiffusionInference.apply_diffusion` (`noise_schedule_benchmark.py:553-590`) consumes Haiku keys in this order:

1. **VE (`diffusion_kind == "edm"`)**: key A -> `diffuse_coordinates_edm` (1 normal, `[N,A,3]`); key B -> `diffuse_sequence` (1 uniform, `[N]`).
2. **VP / VP-fixed / flow**: key A -> `diffuse_atom_cloud` (1 normal `[N,A,3]`; or split into 2 if `correlated_residues`); key B -> `diffuse_sequence` (1 uniform `[N]`).

Note: **both `if` blocks are plain `if`, not `elif`** (`:569` and `:574`), so a hypothetical config with `diffusion_kind == "edm"` that also matched the second tuple would draw four keys. It cannot, but the shape of the code matters if you port more kinds.

Downstream of `apply_diffusion`, per `Diffusion.__call__`: one key for `get_random_neighbours(fape_neighbours=64)` on clean CB (`:1042`, result discarded at inference), then 3 keys per DiffusionBlock x 6 blocks, then 2 keys in `aa_decoder.train` (uniform `t`, then `diffuse_sequence`), then 1 more for that decoder's `get_random_neighbours(0)`. See `notes/01_INFERENCE_PATH.md` §3 — unchanged by this spec, but do not skip these draws if you ever attempt stream-level parity.

`StructureDiffusionNoise` (`noise_schedule_benchmark.py:591-632`) is a byte-identical copy of `apply_diffusion`; the inference-library path therefore has the same draw order. Implement once.

## 2.8 Training-time sigma sampling (for completeness, not needed at inference)

`StructureDiffusion.apply_diffusion` (`:344-410`):
* VE: `get_sigma_edm(batch, meanval=c.pos_mean_sigma, stdval=c.pos_std_sigma, ...)` (`:663-691`) -> `sigma = exp(meanval + stdval * normal)`, one draw per **batch item**, gathered with `[batch]`. Config values are 1.6 / 1.4 (same as `sigma_scale_edm`).
* VP/flow: `t_pos = uniform(key, (N,))[batch]`, then `t_pos = SIGMA_SCALE[c.diffusion_time_scale](t_pos)`; separately `t_seq = uniform(key, (N,))[batch]`. If `"t_pos" in data` both are overridden **after** the draws (`:365-367`, `:392-394`) — the keys are still consumed. That is the same override-after-draw pattern you see at inference.

---

# 3. SAMPLING LOOP

## 3.1 Reference implementation: the eval script

**Implement `eval_noise_schedule_benchmark.py:592-673` as the reference.** It is what the README (`README.md:339`, `:354-358`) documents and what produced the published numbers. `salad/inference/sampling.py` is a later refactor with small divergences (§3.6) and a hard bug.

## 3.2 Full pseudocode (unconditional, `default_vp`, `A = 20`)

```
# --- setup, once per process -------------------------------------------
num_aa, resi, chain, is_cyclic, cyclic_mask = parse_num_aa(opt.num_aa)   # :489
base_chain = chain                                                        # :490
if opt.merge_chains: chain = zeros_like(chain)                            # :491-492
out_steps  = parse_out_steps(opt.out_steps)                               # :493
timescale_pos = parse_timescale(opt.timescale_pos)      # cached; upstream re-parses per step
timescale_seq = parse_timescale(opt.timescale_seq)      # default "1" -> constant 1
cloud_std_fn  = parse_cloud_std(opt.cloud_std)                            # :548
init_pos   = zeros((num_aa, 20, 3), float32)                              # :531
init_aa_gt = full((num_aa,), 20, int32)                                   # :545
init_local = zeros((num_aa, 128), float32)                                # :547

# --- per design --------------------------------------------------------
for idx in range(num_designs):                                            # :556
    cloud_std = cloud_std_fn(num_aa)              # np RNG draw, or None  # :557
    data = dict(
        pos           = init_pos,                 # (L,20,3) f32
        aa_gt         = init_aa_gt,               # (L,)  i32, all 20
        seq           = init_aa_gt,               # (L,)  i32, all 20
        residue_index = resi,                     # (L,)  i32
        chain_index   = chain,                    # (L,)  i32
        batch_index   = zeros_like(chain),        # (L,)  i32
        mask          = ones_like(chain, bool),   # (L,)  bool
        cyclic_mask   = cyclic_mask,              # (L,)  bool
        t_pos         = ones((L,), f32),
        t_seq         = ones((L,), f32),
    )                                                                     # :558-569
    if cloud_std is not None: data["cloud_std"] = cloud_std               # :576-577
    init_prev = dict(pos=zeros_like(init_pos), local=init_local)          # :586-589
    prev = init_prev                                                      # :590

    for step in range(num_steps):                                         # :592
        key, subkey = split(key)                                          # :593
        raw_t  = 1 - step / num_steps            # <-- see 3.3            # :595
        s_pos  = timescale_pos(raw_t)                                     # :597
        s_seq  = timescale_seq(raw_t)                                     # :600
        data["t_pos"] = s_pos * ones_like(data["t_pos"])                  # :599
        data["t_seq"] = s_seq * ones_like(data["t_seq"])                  # :600
        if raw_t < prev_threshold:                                        # :604
            prev = init_prev                     # RESET self-conditioning# :605
        update, prev = model(params, subkey, data, prev)                  # :609
        data["pos"]      = update["pos"]         # x0 prediction, fed back# :610,620
        data["seq"]      = argmax(update["aa"], -1)                       # :621
        maxprob          = softmax(update["aa"], -1).max(-1)              # :622
        data["atom_pos"] = update["atom_pos"]                             # :623
        data["aatype"]   = update["aatype"]                               # :624
        if step in out_steps:                                             # :625
            write_pdb(data, maxprob, base_chain, f"result_{idx}_{step}.pdb")
            if step == out_steps[-1]: break                               # :672-673
```

There is **no integrator**. Each step: set sigma, re-noise the previous x0 prediction *inside* the model (`apply_diffusion`), predict a new x0, feed it back. It is repeated single-shot denoising, not DDPM/DDIM/ODE.

## 3.3 The off-by-one in `raw_t`

`eval_...:595`: `raw_t = 1 - step / opt.num_steps`, with a `FIXME` at `:594` and the alternative `1 - step / (num_steps - 1)` commented out at `:596`.

Consequences for the default `num_steps=500, out_steps=[400]`:
* `step` ranges `0..400` inclusive -> **401 model evaluations**, then `break`.
* `raw_t` ranges `1.0, 0.998, ..., 0.2` (at step 400). It **never reaches 0**.
* The final `sigma` fed to the model under `cosine(t)` is `sigma_scale_cosine(0.2) = 0.320385`, i.e. the last denoising step is still done from a moderately noisy state, and the structure that is written is the x0 prediction at that noise level.
* If a user set `out_steps = 499` (the last step), `raw_t = 1 - 499/500 = 0.002`, `sigma = 0.00314` — still not 0.

**Implement `1 - step / num_steps` exactly.** Use Python float division (`step / num_steps`), not integer or tensor math, so the value matches bit-for-bit. Expose the divisor as a non-default keyword (`_raw_t_denominator=None -> num_steps`) only if you want the fixed variant available for experiments; the default must be the buggy one.

## 3.4 `out_steps`

* Membership test is `step in out_steps` — a list, unsorted, as parsed.
* Termination is `if step == out_steps[-1]: break`, **nested inside** the `if step in out_steps` block (`:672-673`).
* **Latent bug:** `out_steps[-1]` is the *last element as written*, not the maximum. `--out_steps "400,100"` breaks at step 100 and never reaches 400. `--out_steps "100,400"` writes both. Reproduce; optionally emit a warning when `out_steps != sorted(out_steps)`.
* `--out_steps all` -> `range(1000)`, `[-1] == 999 >= num_steps`, so the break never fires and the loop ends naturally at `step = num_steps - 1`.
* If `out_steps` is empty the upstream code would `IndexError` only when a step matched (it never does) — the loop just runs to completion with no output. `inference/sampling.py:90` evaluates `out_steps[-1]` unconditionally every iteration and *would* `IndexError` on an empty list.

## 3.5 `prev_threshold` self-conditioning reset

`eval_...:604-605`:
```
if raw_t < prev_threshold:
    prev = init_prev          # dict(pos=zeros(L,20,3), local=zeros(L,128))
```
* Checked **every** iteration, comparing **`raw_t`**, not `scaled_t`. The `FIXME` at `:601` and the commented-out `scaled_t` version at `:602-603` show the author was unsure; the shipped and published behaviour is `raw_t`.
* With `prev_threshold = 0.8` and `num_steps = 500`: `raw_t < 0.8` from `step = 101` onward, so self-conditioning is live only for steps 0..100 and `prev` is zeroed for every later step. `prev` returned by the model is still *captured* each step (`:609`) but is immediately discarded at the top of the next iteration.
* Effectively: self-conditioning is **on for the first 20% of the trajectory and off for the remaining 80%** at the default.
* `--prev_threshold 0.99` (recommended for `ve_scaled`, README:509-515) -> live for steps 0..4 only.
* `init_prev` must be the *same object* re-assigned (upstream reuses one dict), and must be zeros. Do not lazily recreate it with `torch.zeros_like(prev["pos"])` — that is equivalent, but be careful that `prev["pos"]` shape is `(L, A, 3)` and `prev["local"]` is `(L, c.local_size)` = `(L, 128)`.
* `Diffusion.init_prev` (`noise_schedule_benchmark.py:1120-1126`) returns `0.0 * normal(key, pos_noised.shape)` — a key draw producing exact zeros — but it is only reached when `prev is None`, which never happens in the eval loop. Skip it in torch; note it if you ever chase RNG-stream parity.

## 3.6 `--timescale_pos` / `--timescale_seq`

* `--timescale_pos` default `"cosine(t)"` (`eval_...:457`). Maps `raw_t -> sigma`, written into `data["t_pos"]` as a `(L,)` float32 constant vector.
* `--timescale_seq` default `"1"` (`eval_...:458`). Maps `raw_t -> 1.0` -> `data["t_seq"]` all ones -> every sequence position masked to token 20 every step. `default_vp` has `diffuse_sequence=False` in its config, so `t_seq` never enters the feature stack; it only drives the `diffuse_sequence` RNG draw.
* Both are **inference CLI arguments, not config keys**. `c.diffusion_time_scale = "cosine"` in the config governs *training-time* sigma sampling only.
* Recommended pairings (README:426-433):
  * `default_vp`: `--timescale_pos "cosine(t)" --cloud_std none --prev_threshold 0.8`
  * `default_vp_scaled`: `+ --cloud_std "default(num_aa)"`
  * `default_ve_scaled`: `--timescale_pos "ve(t)" --prev_threshold 0.95..0.99`

## 3.7 Every difference between the eval script and `inference/sampling.py`

| # | eval script (`eval_...:592-673`) | `inference/sampling.py:50-96` | recommendation |
|---|---|---|---|
| 1 | loop `range(num_steps)` | `range(start_steps, num_steps)` — supports resuming/partial diffusion (`:65`) | port the `start_steps` kwarg, default 0 (superset, no behaviour change) |
| 2 | mutates the caller's `data` dict in place | shallow-copies `data` once at `:61` | copy — safer, identical results |
| 3 | `timescale_seq` parsed from `"1"` -> constant 1 | `timescale_seq=None` -> `scaled_t_seq = 1.0` hard-coded (`:69-71`) | identical outcome; accept `None` and `str` |
| 4 | `out_steps` is always a list | accepts `int` or list; `int` -> `[int]` and returns `results[0]` (`:56-60, 92-96`) | accept both |
| 5 | `if step == out_steps[-1]: break` nested inside `if step in out_steps` (`:672`) | `if idx == out_steps[-1]: break` at loop level (`:90`) | functionally identical; use the loop-level form but guard against empty `out_steps` |
| 6 | writes PDB / FASTA inline | returns a list of dicts, no I/O | **split**: sampler returns snapshots, a separate writer does I/O |
| 7 | computes `maxprob = softmax(update["aa"]).max(-1)` at `:622` for b-factors | does not; `to_protein` recomputes it from `data["aa_logits"]`, falling back to **zeros** if absent (`inference/data.py:152-155`) | compute in the writer from `update["aa"]`, and require it |
| 8 | snapshot dict does **not** carry `update` | stores `data["result"] = update` and `data["aa_logits"] = update["aa"]` into each snapshot (`:85-86`) | carry both — needed for b-factors and diagnostics |
| 9 | no latent passthrough | `if "latent" in data and "latent" in update: data["latent"] = update["latent"]` (`:79-80`) | port; harmless |
| 10 | `--sample_final`: at out_steps, calls a second jitted model with `sample_aa=True, temperature=0.1` (`:606-607`, built at `:515-517`) | absent | port as an optional `final_step_fn` hook |
| 11 | `--sym` screw symmetrisation of `pos` after each step (`:611-619`), and `sym_noise` injected into the config (`:504-511`) | absent (there is a separate `inference/symmetry.py`) | out of scope for this spec; leave a `post_step_fn(pos, raw_t) -> pos` hook |
| 12 | `--renoise`: 9 extra passes re-running steps `int(0.6*num_steps)..step` with `prev` forced to `init_prev`, producing extra outputs (`:627-644`) | absent | port only if needed; it is an ablation, not the default |
| 13 | PDB `chain_index` = `base_chain` (pre-`--merge_chains`) (`:665`) | `data["chain_index"]` (post-merge) (`inference/data.py:160`) | use `base_chain`; keep the merged `chain` only inside `data` |
| 14 | `atom37_mask = get_atom37_mask(aatype)` (`:660`) | `data["mask"][:, None] * get_atom37_mask(aatype)` (`inference/data.py:151`) | use the eval form for parity; expose the residue-mask multiply as a flag |
| 15 | `cloud_std` drawn once per design and put into `data` (`:557, 576-577`) | caller's responsibility | keep the eval behaviour in the driver |
| 16 | — | `Sampler.generate` (`:45-48`) references an undefined local `key` -> **`NameError` on first call** | do not port; if you want a generator, thread the key/generator through explicitly |
| 17 | after the first write, `for idr, data in enumerate(results)` at `:645` **rebinds the outer `data`** to the snapshot copy made at `:626` | n/a | benign (shallow copy, identical contents) but do not reproduce the rebinding; it is an accident |

Note: the dict comprehensions at `:626, 643, 645` use `key` as the comprehension variable. Python 3 comprehensions have their own scope, so the PRNG `key` is **not** clobbered. Do not "fix" this as if it were a bug.

## 3.8 Recommended torch API

```python
@dataclass
class SamplerConfig:
    num_steps: int = 500
    out_steps: Sequence[int] = (400,)
    start_steps: int = 0
    prev_threshold: float = 0.8
    timescale_pos: str | Callable = "cosine(t)"
    timescale_seq: str | Callable | None = "1"

def sample(step_fn, data, prev, cfg, *, generator=None,
           post_step_fn=None, final_step_fn=None) -> list[dict]
```
`step_fn(data, prev, generator) -> (update, prev)`. Snapshots are shallow dict copies containing at least `pos, seq, atom_pos, aatype, residue_index, chain_index, mask, t_pos, t_seq, result, aa_logits`.

---

# 4. PDB OUTPUT

## 4.1 `atom14_to_atom37(atom14_data, aatype)` — `salad/aflib/model/all_atom_multimer.py:228-241`

```
assert atom14_data.ndim in (2, 3)
idx = RESTYPE_ATOM37_TO_ATOM14[aatype]                  # (N, 37) int32
atom37 = gather(atom14_data, idx, axis=1)               # per residue: (14,C)[idx] -> (37,C)
mask37 = RESTYPE_ATOM37_MASK[aatype]                    # (N, 37) float32
atom37 = atom37 * (mask37 if ndim==2 else mask37[:, :, None])
```
* Input at the call site is `data["atom_pos"]`, shape `(N, 14, 3)` float32; output `(N, 37, 3)` float32.
* `batched_gather(..., batch_dims=1)` (`aflib/model/utils.py:73-78`) is `vmap(jnp.take(..., axis=0, mode='clip'))` over the residue axis. **`mode='clip'`** clamps out-of-range indices; all indices here are already in `[0, 13]` so clipping never fires, but keep `idx.clamp_(0, 13)` for safety.
* Torch: `atom37 = torch.gather(atom14_data, 1, idx.unsqueeze(-1).expand(-1, -1, 3).long())` then multiply by `mask37.unsqueeze(-1)`. Or advanced indexing `atom14_data[torch.arange(N)[:, None], idx]`.
* The mask multiply **zeroes** the positions of absent atoms (they otherwise hold a duplicate of atom 0 = N, because `RESTYPE_ATOM37_TO_ATOM14` uses 0 as the "not present" sentinel).

## 4.2 `get_atom37_mask(aatype)` — `all_atom_multimer.py:207-208`

```
RESTYPE_ATOM37_MASK[aatype]        # (21, 37) float32 table, gathered -> (N, 37) float32
```
`batched_gather(..., axis=0, batch_dims=0)` = `table[aatype]` with clip. `aatype` at inference is `argmax` over 20 logits so it is always in `[0, 19]`; row 20 (UNK) is all zeros.

## 4.3 Constant tables — port as data, do not retype

Source: `salad/aflib/common/residue_constants.py`. Copy these **five literal tables verbatim** (about 80 lines total) into `salad_torch/io/residue_constants.py`:

| name | line | content |
|---|---|---|
| `atom_types` | `:492-497` | the 37 atom names, order `['N','CA','C','CB','O','CG','CG1','CG2','OG','OG1','SG','CD','CD1','CD2','ND1','ND2','OD1','OD2','SD','CE','CE1','CE2','CE3','NE','NE1','NE2','OE1','OE2','CH2','NH1','NH2','OH','CZ','CZ2','CZ3','NZ','OXT']` |
| `restypes` | `:535-538` | `['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']` (alphabetical by 3-letter code) |
| `restype_1to3` | `:590-646` | 20-entry dict |
| `restype_name_to_atom14_names` | `:505-529` | 21-entry dict (incl. `'UNK'` = 14 empty strings) — this is the atom14 column order, always `N, CA, C, O, CB, ...` |
| `residue_atoms` | `:338-...` | 20-entry dict of heavy atoms present per residue |

Derive the rest at import with the exact upstream builders (do not hand-write them):

| derived | builder | line |
|---|---|---|
| `atom_order = {a: i}` | `residue_constants.py:498` | |
| `restype_order`, `restype_num=20`, `unk_restype_index=20` | `:539-541` | |
| `restype_3to1` | `:647` | |
| `RESTYPE_ATOM37_MASK (21,37) f32` | `_make_restype_atom37_mask` | `all_atom_multimer.py:85-95` |
| `RESTYPE_ATOM14_MASK (21,14) f32` | `_make_restype_atom14_mask` | `:98-109` |
| `RESTYPE_ATOM37_TO_ATOM14 (21,37) i32` | `_make_restype_atom37_to_atom14` | `:112-126` |
| `RESTYPE_ATOM14_TO_ATOM37 (21,14) i32` | `_make_restype_atom14_to_atom37` | `:129-142` |

Builders, restated:
```python
RESTYPE_ATOM37_MASK = zeros((21, 37), f32)
for rt, letter in enumerate(restypes):
    for name in residue_atoms[restype_1to3[letter]]:
        RESTYPE_ATOM37_MASK[rt, atom_order[name]] = 1.0
# row 20 (UNK) stays all zeros

RESTYPE_ATOM14_MASK = [[1. if n else 0. for n in restype_name_to_atom14_names[restype_1to3[rt]]]
                       for rt in restypes] + [[0.] * 14]

RESTYPE_ATOM37_TO_ATOM14 = []
for rt in restypes:
    names = restype_name_to_atom14_names[restype_1to3[rt]]
    n2i = {n: i for i, n in enumerate(names)}     # '' maps to index 4..13 for the pads
    RESTYPE_ATOM37_TO_ATOM14.append([n2i.get(n, 0) for n in atom_types])
RESTYPE_ATOM37_TO_ATOM14.append([0] * 37)

RESTYPE_ATOM14_TO_ATOM37 = [[atom_order[n] if n else 0
                             for n in restype_name_to_atom14_names[restype_1to3[rt]]]
                            for rt in restypes] + [[0] * 14]
```
Caveat in `_make_restype_atom37_to_atom14`: `n2i` is built from the full 14-name list including the `''` pads, so the key `''` maps to the *last* empty slot; since `''` is never in `atom_types` this is unreachable. `n2i.get(n, 0)` is exactly upstream's `atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0`.

**Verification checksums (I computed these from the upstream tables — assert them in a unit test):**
* `RESTYPE_ATOM37_MASK.sum(-1)` per row (21 values) = `[5, 11, 8, 8, 6, 9, 9, 4, 10, 8, 8, 9, 8, 11, 7, 6, 7, 14, 12, 7, 0]`, total `167.0`
* `RESTYPE_ATOM14_MASK.sum(-1)` = the same 21 values
* `RESTYPE_ATOM37_TO_ATOM14.sum()` = `669`
* `RESTYPE_ATOM14_TO_ATOM37.sum()` = `1273`
* `RESTYPE_ATOM37_TO_ATOM14[0]` (ALA) = `[0,1,2,4,3, 0×32]`
* `RESTYPE_ATOM37_TO_ATOM14[17]` (TRP) = `[0,1,2,4,3,5,0,0,0,0,0,0,6,7,0,0,0,0,0,0,0,9,10,0,8,0,0,0,13,0,0,0,0,11,12,0,0]`
* `RESTYPE_ATOM14_TO_ATOM37[0]` (ALA) = `[0,1,2,4,3,0,0,0,0,0,0,0,0,0]`
* `RESTYPE_ATOM14_TO_ATOM37[17]` (TRP) = `[0,1,2,4,3,5,12,13,24,21,22,33,34,28]`

Note the atom37 order puts **CB at index 3 and O at index 4**, whereas atom14 order is `N, CA, C, O, CB` — hence the `[0,1,2,4,3,...]` swap in both maps. The `Protein` docstring at `protein.py:70-71` says "the first three are N, CA, CB" — that comment is **wrong** (it is N, CA, C). Ignore it.

Register the tables as buffers / module-level `torch.tensor`s created once on CPU and `.to(device)` lazily, keyed by device. Keep `atom_types` as a Python list of `str` — the PDB writer iterates it.

## 4.4 `Protein` dataclass — `salad/aflib/common/protein.py:65-98`

Frozen dataclass, **positional field order** (the eval script constructs it positionally at `:662-665`):

| # | field | shape | dtype |
|---|---|---|---|
| 0 | `atom_positions` | `(N, 37, 3)` | float32 |
| 1 | `aatype` | `(N,)` | int (0..20; 20 = 'X') |
| 2 | `atom_mask` | `(N, 37)` | float |
| 3 | `residue_index` | `(N,)` | int (cast to int32 in `to_pdb`) |
| 4 | `chain_index` | `(N,)` | int (cast to int32) |
| 5 | `b_factors` | `(N, 37)` | float |

`__post_init__` raises if `len(unique(chain_index)) > 62`.

Port as a plain `@dataclass` of numpy arrays in `salad_torch/io/pdb.py`. Convert torch -> numpy (`.detach().cpu().numpy()`) at the boundary; `to_pdb` is pure Python string formatting.

## 4.5 `to_pdb(prot)` — `protein.py:238-315`

Exact algorithm:

```python
restypes_x = restypes + ['X']                      # 21 entries
res_1to3 = lambda r: restype_1to3.get(restypes_x[r], 'UNK')    # r==20 -> 'UNK'

if np.any(aatype > 20): raise ValueError('Invalid aatypes.')

PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'  # :36, 62 chars
chain_ids = {i: PDB_CHAIN_IDS[i] for i in np.unique(chain_index)}   # sorted; raise if i >= 62

lines = ['MODEL     1']                            # literal, 5 spaces before the 1
atom_index = 1
last_chain_index = chain_index[0]

for i in range(N):
    if last_chain_index != chain_index[i]:         # chain boundary -> TER for the PREVIOUS residue
        lines.append(_chain_end(atom_index, res_1to3(aatype[i-1]),
                                chain_ids[chain_index[i-1]], residue_index[i-1]))
        last_chain_index = chain_index[i]
        atom_index += 1                            # the TER record consumes a serial number

    res_name_3 = res_1to3(aatype[i])
    for atom_name, pos, mask, b_factor in zip(atom_types, atom_positions[i],
                                              atom_mask[i], b_factors[i]):
        if mask < 0.5: continue                    # strict: 0.5 threshold, not != 0
        record_type = 'ATOM'
        name = atom_name if len(atom_name) == 4 else f' {atom_name}'
        alt_loc = ''; insertion_code = ''; occupancy = 1.00
        element = atom_name[0]                     # first char; valid for C/N/O/S only
        charge = ''
        lines.append(
            f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
            f'{res_name_3:>3} {chain_ids[chain_index[i]]:>1}'
            f'{residue_index[i]:>4}{insertion_code:>1}   '
            f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
            f'{occupancy:>6.2f}{b_factor:>6.2f}          '
            f'{element:>2}{charge:>2}')
        atom_index += 1

lines.append(_chain_end(atom_index, res_1to3(aatype[-1]),
                        chain_ids[chain_index[-1]], residue_index[-1]))
lines.append('ENDMDL')
lines.append('END')
lines = [l.ljust(80) for l in lines]
return '\n'.join(lines) + '\n'
```

`_chain_end` (`protein.py:231-235`) — note the **six literal spaces**:
```python
f'{"TER":<6}{atom_index:>5}      {end_resname:>3} {chain_name:>1}{residue_index:>4}'
```

Formatting rules to get exactly right:
* Atom ordering **within a residue is atom37 order** (`atom_types`), not atom14 order. Absent atoms are skipped by the mask, so a real residue emits its heavy atoms in the order `N, CA, C, CB, O, CG, ...`. That means **CB is written before O**, which is unusual but is what upstream produces.
* `occupancy` is the Python float `1.00`, formatted `{:>6.2f}` -> `"  1.00"`.
* The 10 spaces before `{element:>2}` are literal (columns 67-76 blank).
* `record_type` is always `'ATOM'`; no HETATM, no element-symbol lookup table.
* Every line, including `MODEL     1`, `TER`, `ENDMDL`, `END`, is `ljust(80)`.
* Exactly one `MODEL`/`ENDMDL` pair per file, with a literal model number of 1.
* A final `TER` is always emitted, even for single-chain output.
* `residue_index` and `chain_index` are cast with `.astype(np.int32)` before use (`:255-256`).

## 4.6 The eval-script writer — `eval_noise_schedule_benchmark.py:656-671`

```python
pdb_path     = f"{out_path}/result_{idx}_{step}.pdb"            # :656
atom37       = atom14_to_atom37(data["atom_pos"], data["aatype"])   # (N,37,3)   :659
atom37_mask  = get_atom37_mask(data["aatype"])                      # (N,37) f32 :660
residue_index = np.array(data["residue_index"])                     # :661
protein = Protein(np.array(atom37),                 # atom_positions   :662
                  np.array(data["aatype"]),         # aatype
                  np.array(atom37_mask),            # atom_mask
                  residue_index,                    # residue_index
                  np.array(base_chain),             # chain_index   <- NOT data["chain_index"]
                  maxprob[:, None] * np.array(np.ones_like(atom37_mask, dtype=jnp.float32)))
pdb_string = to_pdb(protein)                                        # :669
open(pdb_path, "wt").write(pdb_string)                              # :670-671
```

### b-factors and the "double softmax"
`maxprob = jax.nn.softmax(update["aa"], axis=-1).max(axis=-1)` (`eval_...:622`), and `update["aa"]` is already `jax.nn.log_softmax(logits)` (`noise_schedule_benchmark.py:1564`, inside `AADecoder.train`).

**This is mathematically an identity, not a bug:** `softmax(log_softmax(x)) = exp(log p_i) / sum_j exp(log p_j) = p_i / 1 = p_i`. So the b-factor is exactly the max predicted amino-acid probability, in `[0, 1]`, rendered as e.g. `"  0.87"`. Reproduce it by applying `torch.softmax(aa_log_probs, dim=-1).max(dim=-1).values` (same code path, same float32 rounding) rather than `aa_log_probs.exp().max(...)` — the results differ in the last ulp.

`b_factors` shape is `(N, 1) * (N, 37) = (N, 37)`: the same scalar repeated across all 37 atom slots, **unmasked** (the `np.ones_like(atom37_mask)` factor is all ones, not the mask). `to_pdb` only reads slots whose `atom_mask >= 0.5`, so the unmasked values never surface.

`maxprob` is the value from the **current** step (recomputed at `:622` each iteration), so it always matches the structure being written.

### Residue numbering and chain gaps
* `residue_index` comes straight from `parse_num_aa`: chain 0 is `0, 1, ..., n0-1`; chain k starts at `(last residue index) + 50`. **Numbering starts at 0**, which is legal-ish but unusual for PDB (most tools expect 1-based). Reproduce; do not add 1.
* `chain_index` is `base_chain` — the pre-`--merge_chains` chain assignment (`eval_...:490`), so a merged run still writes separate PDB chains with `TER` records between them, even though the model saw one chain.
* With `--dssp` or `--template` overrides, `resi` is replaced with `np.arange(num_aa)` and `base_chain` with zeros / the template's chain index (`eval_...:524-530`).

### aatype
`update["aatype"]` is `argmax(aa_logits, -1)` over **20** classes (`noise_schedule_benchmark.py:1104`), so it is always `0..19` — never 20/UNK — unless `aa_condition` overrides it (`:1106-1109`) or `c.sample_aa` is set. `to_pdb`'s `aatype > 20` check therefore never fires.

### atom_pos provenance (context for the writer)
`data["atom_pos"]` is `(N, 14, 3)`: indices 0-3 (`N, CA, C, O`) are taken from the diffused point cloud, indices 4-13 are rebuilt from predicted torsion angles and ideal geometry (`get_angle_positions`, `noise_schedule_benchmark.py:1110-1117`, and `notes/01_INFERENCE_PATH.md` §5). If `c.encode_atom14` the whole thing is just `pos`.

## 4.7 Recommended torch-side API

```python
# salad_torch/io/atom_layout.py
def atom14_to_atom37(atom14: Tensor, aatype: Tensor) -> Tensor      # (N,14,C)->(N,37,C)
def atom37_to_atom14(aatype, atom37, atom37_mask) -> tuple[Tensor, Tensor]
def get_atom37_mask(aatype: Tensor) -> Tensor                        # (N,37) f32
def get_atom14_mask(aatype: Tensor) -> Tensor                        # (N,14) f32

# salad_torch/io/pdb.py
@dataclass(frozen=True)
class Protein: atom_positions, aatype, atom_mask, residue_index, chain_index, b_factors
def to_pdb(prot: Protein) -> str
def snapshot_to_protein(snapshot, *, chain_index=None, apply_residue_mask=False) -> Protein
def write_pdb(path, snapshot, *, chain_index=None) -> None
```
`snapshot_to_protein` defaults must reproduce the eval script: `chain_index = base_chain` supplied by the caller, `apply_residue_mask=False` (i.e. `get_atom37_mask(aatype)` alone, **not** multiplied by `data["mask"]` — that is the `inference/data.py:151` variant).

---

# 5. Consolidated bug/quirk list to reproduce knowingly

1. `raw_t = 1 - step / num_steps` — off by one; last t is 0.2, never 0. `FIXME` at `eval_...:594`.
2. `out_steps[-1]` is positional, not max — an unsorted `--out_steps` truncates the run. `eval_...:672`.
3. `prev_threshold` compares `raw_t`, not `scaled_t`; `FIXME` at `eval_...:601`. Self-conditioning is off for ~80% of the default trajectory.
4. `position_std` has no `clamp(min=0)` before `sqrt` — NaN for degenerate segments. `diffusion.py:65-67`.
5. `position_std` does `ca -= center` on what is a *view* in torch. Must clone.
6. `index_mean` does `data *= weight` in place (`geometry.py:583`) — already handled in `salad_torch/geometry/segment.py`; do not regress.
7. The two `if` blocks in `apply_diffusion` are not `elif` (`noise_schedule_benchmark.py:569, 574`).
8. `flow` selection differs between training (`in ("flow","flow_scaled")`, `:401`) and inference (`== "flow"`, `:580`).
9. `"flow_scaled"` missing from the VP time-embedding tuple (`:1153`).
10. `diffusion.py:109` `target = target` is a dead statement.
11. `Sampler.generate` (`inference/sampling.py:45-48`) uses an undefined `key` -> `NameError`. Do not port.
12. `inference/sampling.py:90` indexes `out_steps[-1]` unconditionally -> `IndexError` on an empty list.
13. The b-factor "double softmax" (`eval_...:622` over already-log-softmaxed `aa`) is an exact identity, not a bug. Keep the same code path for bit parity.
14. `Protein.atom_positions` docstring claims the first three atoms are "N, CA, CB" (`protein.py:70-71`); they are N, CA, C.
15. PDB residue numbering starts at 0 with +50 inter-chain gaps, and atoms are written in atom37 order so CB precedes O.
16. The eval script's `for idr, data in enumerate(results)` at `:645` rebinds the outer `data`. Benign; do not replicate.
17. `cloud_std_default` uses numpy's global RNG, unseeded by `--jax_seed`, so `--cloud_std "default(num_aa)"` runs are not reproducible from the jax seed alone.
18. `eval_...:597, 600` re-parse and re-`eval` the timescale strings on every step (500x per design). Cache instead.
