# SALAD inference path (JAX reference) — spec for the PyTorch port

Reference clone: `upstream/salad` @ 8348117. All line numbers refer to it.

## 1. Entry points

**The script the README/manuscript used** (`README.md:339`):
`salad/training/eval_noise_schedule_benchmark.py` — 675 lines, no `main()`, all
logic under `if __name__ == "__main__":` at `:451`. Run as a file path.

```bash
python salad/training/eval_noise_schedule_benchmark.py \
    --config default_vp --params params/default_vp-200k.jax \
    --out_path out/ --num_aa 100 --num_steps 500 --out_steps 400 \
    --num_designs 10 --prev_threshold 0.8 --timescale_pos "cosine(t)"
```

**Newer library API**: `salad/inference/*`, demonstrated in
`examples/inference_api.py`. Numerically equivalent for the default configs —
`StructureDiffusionNoise` is a byte-identical copy of
`StructureDiffusionInference.apply_diffusion`, and `prepare_condition` consumes
zero RNG keys at inference, so the draw order matches.
`salad/inference/shape.py` is an empty stub; `Sampler.generate`
(`sampling.py:45`) is buggy (undefined `key`).

**Port against the eval script** — it is what the published numbers came from.

## 2. The sampling loop

There is **no explicit DDPM/ODE integrator**. Each step is pure x0-prediction
with re-noising:

```python
prev = dict(pos=zeros(L,20,3), local=zeros(L,128))
for step in range(num_steps):                 # 500
    key, subkey = split(key)                  # :593  one key per step
    raw_t    = 1 - step / num_steps           # :595  NOTE: /num_steps, not /(num_steps-1)
    data["t_pos"] = timescale_pos(raw_t) * ones(L)     # cosine(t)
    data["t_seq"] = timescale_seq(raw_t) * ones(L)     # default 1.0
    if raw_t < prev_threshold:                # :604
        prev = init_prev                      # :605  RESET self-conditioning
    update, prev = model(params, subkey, data, prev)   # :609
    data["pos"]      = update["pos"]          # :620  x0 prediction fed straight back
    data["seq"]      = argmax(update["aa"])
    data["atom_pos"] = update["atom_pos"]
    data["aatype"]   = update["aatype"]
    if step in out_steps: write_pdb(); break_if_last()
```

Noise is applied **inside** the model each step (`noise_schedule_benchmark.py:456`
→ `:578`), via `diffuse_atom_cloud` (`utils/diffusion.py:69-114`):

```
target   = cloud_std * N(0,I)                      # (L, A, 3)
target  -= mean over residues of target[:,1]       # CA-centred, per batch_index
x_noised = x_prev * sqrt(1 - t**2) + target * t    # VP   (diffusion.py, flow=False)
x_noised = x_prev + t * centred_N(0,I)             # VE   (diffusion_kind=="edm")
```

`cloud_std = sigma_data = 10.0` for `default_vp` (`diffusion_kind="vpfixed"`);
for `default_vp_scaled` (`"vp"`) it comes from `--cloud_std` or the per-structure
CA std.

Within `Diffusion.__call__`, 6 `DiffusionBlock`s refine positions by a
frame-local additive update (`:2234` → `update_positions`, `:2740`):

```
frames, local_pos = extract_aa_frames(pos)
pos_update = out_scale * Linear(A*3, w_init=zeros, no bias)(local_norm)
pos        = frames.apply_to_point(local_pos + pos_update)
```

`out_scale` from `preconditioning_scale_factors` (`:967`): VP →
`{in:1, out:sigma_data, skip:1}`; VE → `edm_scaling` (`:940`).

**Step count**: `--num_steps 500 --out_steps 400` ⇒ 401 model evaluations
(step 0..400), then break. `raw_t` runs 1.0 → 0.2. It never reaches 0.

**Timescales** (`parse_timescale`, `eval_...:82`) `eval()` the string with `t`,
`cosine`, `framediff`, `ve`, `edm` in scope. `cosine(t)` =
`sigma_scale_cosine` (`noise_schedule_benchmark.py:62`):
`alpha_bar = cos((t+s)/(1+s)·π/2)² / cos(s/(1+s)·π/2)²`, `s=0.01`, clipped to
[0,1]; `sigma = sqrt(1-alpha_bar)`. `t=1 → 1.0`, `t=0.2 → 0.3203`.

**Self-conditioning**: `prev` is reset to zeros whenever `raw_t < prev_threshold`,
re-checked every iteration — so with the default 0.8 it is active only for steps
0..100 of 500 and zeros thereafter. A `FIXME` at `:601` notes the author was
unsure whether to compare `scaled_t` or `raw_t`; shipped behaviour uses `raw_t`.
`prev["local"]` enters at `:1160`; `prev["pos"]` is the second attention target
in every block (`:2192`, `:2246`).

## 3. RNG / stochasticity  (the hard part for parity)

One host key split per step. Draw order inside one call (default_vp):

1. `prepare_condition` — **0 keys** (short-circuits on `set_condition`, `:2562`).
2. `apply_diffusion` — key 1 `diffuse_atom_cloud`; key 2 `diffuse_sequence`.
3. `Diffusion.__call__`:
   - key 3: `get_random_neighbours(fape_neighbours=64)` on clean positions for
     distogram supervision (`:1042`) — unused at inference, still burns a key.
   - `diffusion_stack`: `layer_stack` draws one outer key then splits per layer
     (`aflib/model/layer_stack.py:147`). Each of 6 blocks draws 3 keys:
     `extract_neighbours` Gumbel-top-k (`utils/geometry.py:289`), plus 2 in
     `extract_condition_neighbours` (`:1819`, `:1825`).
   - `aa_decoder.train` (`:1548`): uniform `t`, `diffuse_sequence`, then one more
     for `get_random_neighbours(0)`.
   - `Diffusion.init_prev` (`:1120`) draws a key but is never reached.

Output-changing randomness = (a) diffusion noise, (b) **Gumbel-top-k random
neighbour selection in every block**. (b) blocks bit-exact parity unless we
replicate threefry + the exact Gumbel-top-k. **For debugging, feed a fixed
neighbour list**; otherwise accept distributional parity only.

## 4. Featurisation, unconditional run of length L

`A = 14 if encode_atom14 else 5 + augment_size` → **A = 20** for default_vp.
`local_size = 128`.

| key | shape | dtype | construction |
|---|---|---|---|
| `pos` | (L,20,3) | f32 | zeros at step 0, then previous `update["pos"]` |
| `aa_gt` | (L,) | i32 | all 20 (= MASK; 0..19 are AAs) |
| `seq` | (L,) | i32 | `aa_gt`, then `argmax(update["aa"])` |
| `residue_index` | (L,) | i32 | chain 0 → `0..n0-1`; each next chain starts at `last+50` |
| `chain_index` | (L,) | i32 | 0,1,…; all zeros if `--merge_chains` |
| `batch_index` | (L,) | i32 | zeros (one structure per call) |
| `mask` | (L,) | bool | ones |
| `cyclic_mask` | (L,) | bool | all False unless chain written `cN` |
| `t_pos`, `t_seq` | (L,) | f32 | set each step |

`prev = dict(pos=zeros(L,20,3), local=zeros(L,128))`.

Optional and **absent** unconditionally: `cloud_std`, `dssp_mean`,
`dssp_condition`, `dmap`, `omap`, `dmap_mask`, `aa_condition`, `chain_contacts`.

Sentinels materialised in `prepare_condition` (`:482-547`): `aa = 20`,
`dssp = 3`, `dssp_mean = 0`, masks all False, `dmap/omap = 0`,
`flags = concat([chain_contacts[...,None], zeros(L,L,2)])`.
`Condition` (`:2521`) sums three masked terms, all masked off ⇒ **`condition` is
exactly zero (L,128)**. Do not shortcut it: the outer LayerNorm at `:1170` maps
0 → the learned `offset` of `~prepare_features/layer_norm_2`.

`Diffusion.prepare_features` (`:1128`):
```
_, local_pos = extract_aa_frames(pos_noised)                   # (L,20,3) frame-local
time_emb   = concat([t_pos[:,None], fourier_time_embedding(t_pos, 256)])   # (L,257)
local_feats= concat([time_emb, local_pos.reshape(L,60),
                     distance_rbf(|local_pos|,0,22).reshape(L,1280),
                     LayerNorm(prev["local"])])                # 257+60+1280+128 = 1725
local      = LayerNorm(MLP(1725->512->128)(local_feats))
```
Checkpoint confirms `~prepare_features/mlp/linear.w = (1725,512)`, so the RBF
block is 20 atoms × **64** bins. `diffuse_sequence=False` for default_vp, so the
sequence time-embedding and `one_hot(seq,21)` are not concatenated.

**The Encoder is not used at inference.** `StructureDiffusionInference.__call__`
never calls `prepare_data`/`encode`; `~encode/encoder/*` is dead weight for pure
generation (it matters only for partial diffusion / autoencoder paths). The 15
extra pseudo-atoms start at zero and are diffused like the backbone.

## 5. Outputs and PDB writing

`update` (from `Diffusion.__call__`, `:1076`):

| key | shape | note |
|---|---|---|
| `pos` | (L,20,3) | denoised cloud; first 5 ≈ N,CA,C,O,CB; **fed back** |
| `local` | (L,128) | fed back as `prev["local"]` |
| `aa` | (L,20) | **log-softmax**, not logits (`:1555`) |
| `aatype` | (L,) | `argmax(aa)` unless `sample_aa` |
| `atom_pos` | (L,14,3) | atom14 all-atom |
| `trajectory` | (6,L,20,3) | per-block |

plus diagnostics (`distogram`, `dssp`, `raw_angles`, …).

`atom_pos` (`get_angle_positions`, `:1435`): a 256-wide MLP predicts 7×(sin,cos)
torsions, normalised, then `single_protein_sidechains` builds atom14; finally
```
atom_pos = concat([pos[:, :4, :], angle_pos[:, 4:, :]], -2)
```
i.e. **N/CA/C/O come from the diffused cloud; CB onward is rebuilt from
predicted χ angles + ideal geometry**.

PDB writing (`:656-671`): `atom14_to_atom37` → `Protein(...)` → `to_pdb`
(`aflib/common/protein.py:238`). B-factors = `softmax(update["aa"]).max(-1)`
where `aa` is *already* log-softmaxed — a harmless double softmax. Residue
numbering starts at **0**, +50 gap between chains.

## 6. Checkpoint format

- Loader: `pickle.load` at `eval_...:519` and `inference/model.py:10`.
- Plain **pickle protocol 4**, uncompressed. Flat
  `dict[module_path, dict[param_name, jax.Array]]` — the standard Haiku tree.
- Arrays pickle via `jax._src.array._reconstruct_array`, which just wraps a
  numpy array — so a custom `Unpickler` with a stubbed `find_class` reads them
  with the stdlib alone. (We instead convert once inside the container; see
  `00_SETUP.md`.)
- **The config is not stored in the file.** It is selected by name from
  `salad/modules/config/noise_schedule_benchmark.py`.

`dotdict` (`utils/collections.py:5`) sets `__getattr__ = dict.get`, so **any
unset option silently reads `None`**. That is how `c.minimal`, `c.atom`,
`c.multi_motif`, `c.sym_noise`, `c.symm`, `c.sample_aa`, `c.refine`,
`c.use_omap`, `c.aa_trajectory`, `c.block_symmetry`, `c.nonequivariant_dense`,
`c.repeat`, `c.encode_atom14`, `c.linear_aa`, `c.equivariance` all evaluate
false for `default_vp`. `config.eval = True` is set by every eval script but is
**never read** in `noise_schedule_benchmark.py`.

`default_vp` (`config/noise_schedule_benchmark.py:3-59`): `augment_size=15,
encoder_depth=2, local_size=128, pair_size=64, num_neighbours=48,
diffusion_depth=6, block_size=1, heads=8, key_size=32, factor=4,
relative_position_encoding_max=32, diffusion_kind="vpfixed", sigma_data=10.0,
preconditioning="vp", diffusion_time_scale="cosine", diffuse_sequence=False,
aa_decoder_kind="adm", aa_decoder_depth=3, time_embedding=True, resi_dual=False,
fape_neighbours=64`.

Variants: `default_vp_scaled` = `+diffusion_kind="vp"` (needs `--cloud_std`);
`default_ve_scaled` = `preconditioning="edm_scaled"`, `diffusion_kind="edm"`;
`multimotif_vp` = `+multi_motif=True`.

**Haiku conventions**: `Linear` stores `w` as `(in, out)` — the **transpose** of
`torch.nn.Linear.weight`. `LayerNorm` stores `scale`/`offset`. `layer_stack`
params are indexed `[layer_idx]` on axis 0.

## 7. flexcraft / flexloop — reuse assessment

Worth copying (thin, no JAX math): `flexcraft/utils/options.py:23` and
`flexloop/flexloop/utils.py:28` (identical argparse wrappers — inline the 30
lines and drop the flexloop dependency entirely); `flexcraft/files/pdb.py:11-99`
(pure file I/O); `flexcraft/utils/io.py` glue; `flexcraft/utils/rng.py:10`
(8-line keygen). `flexcraft/data/data.py:50-523` `DesignData` is a good API
model — `to_salad` (`:93`) is the canonical atom14 → salad `pos` converter
(`positions_to_ncacocb`, then pad to `5+augment_size` by repeating CA) — but it
is jnp throughout, so reimplement rather than import.

Do **not** reuse: `flexcraft/structure/_colab_af/*` (vendored 28 kLOC AlphaFold),
`flexcraft/sequence/mpnn.py` (627-line JAX ProteinMPNN), `structure/{boltz,
protenix,af3_like}`, `rosetta/*`.

`python -m flexcraft.pipelines.denovo` drags in pyrosetta, AF2 params and
ProteinMPNN — use it as a worked example only. Its salad part is just lines
26-56, 107-118, 129-141.

## 8. Parity checklist / gotchas

1. `raw_t = 1 - step/num_steps` (not `num_steps-1`); last `t` at `out_steps=400`
   is 0.2 (σ=0.3203). There is a `FIXME` at `:594`.
2. Noise is applied *inside* the model on the previous x0 prediction, fresh key
   each step. Re-noising is not a residual/ancestral update.
3. Noise is CA-mean-centred per `batch_index` (`diffusion.py:100`), not
   zero-mean per atom.
4. Self-conditioning is off for ~80 % of the trajectory at the default threshold.
5. `condition` is exactly zero unconditionally, but the LayerNorm still maps it
   to a learned offset. Don't optimise it away.
6. `aa_gt` is all-20 and `aa_decoder.train` (not `.argmax`) runs at inference, so
   the ADM masking RNG advances. Skipping it shifts the RNG stream.
7. Distogram + `distogram_neighbours` + dssp head are computed every step and
   discarded; dropping them changes the RNG stream by one key.
8. `result["aa"]` is log-probabilities, not logits.
9. PDB residue indices start at 0, +50 between chains.
10. `inference/sampling.py` and the eval script diverge on `cloud_std`, symmetry,
    `renoise`, `base_chain` vs `chain_index`, and the atom37 mask. The eval
    script is the reference for published results.
