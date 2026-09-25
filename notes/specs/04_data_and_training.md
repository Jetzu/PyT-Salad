# PyTorch port specification: data pipeline and training loop

Reference (read-only): `/home/hpc/v181bb/v181bb10/SALAD_torch/upstream/salad`,
`/home/hpc/v181bb/v181bb10/SALAD_torch/upstream/flexloop`.
Target: `/home/hpc/v181bb/v181bb10/SALAD_torch/salad_torch`.
Companion specs: `notes/specs/02_schedules_noising_sampling_pdb.md` (noise
schedules, forward noising, sampling, PDB I/O — do not duplicate),
`notes/specs/03_training_losses.md` (the loss function this training loop
calls — do not duplicate), `notes/04_PARITY_AND_PERF.md` (§B1 is load-bearing
for this spec, see §A.8).

This spec is written for **`default_vp`** specifically (config at
`salad/modules/config/noise_schedule_benchmark.py:3`) and for the training
script actually used to produce it, **`salad/training/train_noise_schedule_benchmark.py`**.
Every fact below states whether it is what actually runs for `default_vp`, or
an alternate/dead code path kept for other configs.

---

## 0. Headline findings the writer must not miss

**(a) Data augmentation is OFF for `default_vp`.** `StructureDiffusion.prepare_data`
and `.encode` both gate their rotation/translation augmentation on
`c.equivariance == "semi_equivariant"` (`noise_schedule_benchmark.py:131,
176, 202, 215`). `default_vp` never sets `equivariance`, so `dotdict.__getattr__
= dict.get` (see `03_training_losses.md` §0a) makes `c.equivariance` evaluate
to `None` → the branch never fires. **The only geometric transform ever
applied to training coordinates is a one-time re-centering** (subtract the
per-batch-item mean CA position, `prepare_data` step 4, see §A.3). There is
no random rotation, no random translation beyond centering, and no reflection
augmentation anywhere in the `default_vp` data path. This is consistent with
an SE(3)-equivariant architecture that does not need pose augmentation — do
not add any.

**(b) `config.min_size` / `config.max_size` are dead configuration keys.**
`default_vp` declares `min_size=50, max_size=None`
(`noise_schedule_benchmark.py` config, "dataset constraints" comment) but
**no code anywhere in the repository reads `c.min_size` or `c.max_size`**
(grepped, zero hits). The actual size-bucketing values used at training time
are hardcoded as **literal keyword arguments in the training script**,
`train_noise_schedule_benchmark.py:160-167`: `min_size=16, max_size=opt.num_aa`
(CLI default `opt.num_aa=1024`). Port these as training-script/CLI parameters,
not as fields the model config forwards.

**(c) `blocks` / `block_adjacency` / `drop_dssp` are entirely dead for
`default_vp`'s loss and conditioning.** `prepare_condition`
(`noise_schedule_benchmark.py:223-258`) calls `assign_dssp` and then
`drop_dssp`, but:
```python
dssp, blocks, block_adjacency = assign_dssp(pos, data["batch_index"], pos_mask.any(axis=-1))
_, block_adjacency = drop_dssp(hk.next_rng_key(), dssp, blocks, block_adjacency, p_drop=c.p_drop_dssp)
```
The `_` discards `drop_dssp`'s **dropped** copy of `dssp` — the loss and
`Condition` module both consume the **original, undropped** `dssp` (the first
return value of `assign_dssp`, never reassigned). The reassigned local
`block_adjacency` (dropped-out) is **never referenced again anywhere** —
not stored in the `result` dict returned by `prepare_condition`, not passed
to `Condition(...)` (whose call signature at line 242 only takes `aa, dssp,
mask, residue_index, chain_index, batch_index`). So: **implement `assign_dssp`
fully (its first return value, `dssp`, is required), but `blocks`,
`block_adjacency`, and the whole of `drop_dssp` are provably unreachable dead
code for `default_vp`.** Do not spend effort vectorizing/optimizing them
beyond what's needed to get `dssp` right, and do not wire a dropped
adjacency matrix into conditioning — there isn't one. (Other salad configs,
e.g. SS/ADJ-conditioned variants, likely do consume `block_adjacency`; if you
ever port one of those, revisit this.) `drop_dssp` still **consumes one RNG
key** (`hk.next_rng_key()` at line 239) even though its output is discarded —
reproduce the draw if you need bit-exact RNG-stream parity with JAX, skip it
otherwise.

**(d) The actual training dataset is `BatchedProteinPDBStream` wrapping
`ProteinPDB`.** `train_noise_schedule_benchmark.py`'s `if/elif/else` on
`opt.dataset` (`:132-167`) has three branches — `SYNTHETE`
(`salad/data/synthete.py`), `afdb*` (`salad/data/afdb.py`, whose module
docstring literally says **"Not used in the manuscript"**), and the `else`
branch, which is what `default_vp` actually ran: `BatchedProteinPDBStream`
from `salad/data/allpdb.py`. Sections A.1-A.4 below specify that path in
full; A.5 covers the alternates briefly for completeness, flagged not-in-use.

**(e) EMA is computed one optimizer step stale.** In
`make_training_inner` (`train_noise_schedule_benchmark.py:38-48, 55-56`):
```python
aux_state = ema_step(loop_state.params, loop_state)   # uses params BEFORE this step's update
_, loggables, params, opt_state = update(loop_state.params, loop_state.opt_state, subkey, item)  # produces the NEW params
```
`ema_step` blends `loop_state.aux_state["ema_params"]` with
`loop_state.params` — the parameters **as they were before this step's
gradient update**, not the freshly-updated ones. So at any given step `N`,
the saved `ema_params` reflects `ema_weight * ema_{N-1} + (1-ema_weight) *
params_{N-1}` (params from the *end of step N-1*), and never directly
incorporates the params produced by step `N` until step `N+1` runs. This is
an off-by-one that is immaterial over hundreds of thousands of steps but
means EMA and raw checkpoints saved at the *same* `step_id` are not
"EMA of exactly these params" — reproduce faithfully unless told otherwise.
See §B.4 for the full EMA spec (decay = 0.999, confirmed).

**(f) `rebatch` / `accumulate` / `multigpu` are a 3-level JAX vectorization
stack, not a torch-relevant batching scheme — but `batch_index` semantics
depend on understanding it.** A single `DataLoader` draw of size
`rebatch * accumulate * NUM_DEVICES * num_aa` residues (see §A.8) gets
split, in order, by `pmap` (device axis), then `jax.lax.scan`
(`accumulate` axis, gradient accumulation), then `hk.vmap` (`rebatch` axis).
**Each of the resulting atomic chunks is exactly one `num_aa`-residue,
already-padded dataset draw**, processed as a fully independent forward/
backward pass with its **own local `batch_index` numbering starting at 0**.
`batch_index` is therefore only required to be unique *within one dataset
draw* — it is never required to be globally unique across concatenated
draws, because draws are never algebraically mixed into one flat tensor with
shared segment-op indices; they are always separated back out (by reshape)
before any segment op runs. **A torch port must preserve this separation.**
If you implement gradient accumulation, do it as `K` independent
forward/backward calls (each on its own `num_aa`-sized flat batch with local
`batch_index`, gradients averaged), never by concatenating `K` draws into one
super-flat tensor and hoping `batch_index` collisions don't matter — they
will, silently merging unrelated proteins from different draws that happen
to reuse the same local index. Everything about `rebatch`/`accumulate`/
`multigpu`/`pmap` is **cluster-scaling machinery, incidental to a
single-GPU torch port** — default values are `rebatch=1, accumulate=1`, so
for a straightforward single-device port this whole stack collapses to "one
dataset draw = one forward/backward pass" and can be ignored entirely except
for the batch_index-locality lesson above.

---

# PART A: THE DATA PIPELINE

## A.1 On-disk format and required preprocessing (what must already exist)

This is a two-stage pipeline: (1) a one-time mmCIF→npz conversion
(`salad/data/convert.py`, needs `gemmi` + a full local PDB mirror + a
Chemical Component Dictionary), producing per-structure `.npz` caches; (2) a
clustering/indexing stage (mmseqs2, external to this repo, producing `.tsv`
cluster files) plus a lightweight in-process index built once at `Dataset`
construction time (`compute_index`, pure Python/numpy, fast). **Stage (1) is
incidental infrastructure** — a large, one-time ETL job requiring the full
PDB mirror; port it only if you need to regenerate caches from scratch.
**Reusing the JAX pipeline's existing `.npz` caches directly is the
recommended path** and is what the rest of this spec assumes. Stage (2)'s
index-building (`compute_index`/`compute_asym_index`, `allpdb.py:1429-1493`)
is cheap, pure-Python, and should be ported as-is (§A.2).

### A.1.1 The `.npz` per-structure cache (from `convert.py`)

`parse_structure` (`convert.py:100-288`) parses one mmCIF/PDB file with
`gemmi`, resolving each residue's reference atom order/bonds from a preloaded
Chemical Component Dictionary (`read_components`/`get_residue_info`,
`:25-97`), then bundles per-residue atoms into a dense **"atom24"** layout via
`to_atomX(parse_result, 24)` (`:290-303`):
```python
in_mask = arange(24)[None,:] < per_residue_atom_count[:,None]   # (n_res, 24) bool, front-packed
resdict[field][in_mask] = flat_per_atom_values[field]            # scatter into a dense (n_res, 24, ...) array
```
Atoms within a residue are inserted **in the order the CCD lists them**,
filtered to atoms actually observed (or, for AA/DNA/RNA residues only, a
zero-position/zero-mask placeholder for atoms declared by the CCD but not
resolved in this structure — `convert.py:209-224`); METAL/SMOL residues never
get placeholder atoms. For the 20 canonical amino acids, standard CCD entries
list backbone-then-sidechain in the order N, CA, C, O, CB, ... — **this
must coincide with AlphaFold's canonical atom14 ordering** (`N, CA, C, O,
CB, ...`, see `02_schedules_noising_sampling_pdb.md` §4.3) for the `x` /
`rotation` / `violation` losses (`03_training_losses.md` §7.5-7.10) to compare
the right atoms against the right bond-length/mask tables. **If you ever
regenerate `.npz` caches with a new converter, verify this ordering against
`salad_torch/io/residue_constants.RESTYPE_NAME_TO_ATOM14_NAMES` per residue
type — a silent mismatch would corrupt the all-atom losses without any error.**
If you reuse the existing caches, this is moot; it is a property of data
already on disk, not of anything the training-loop port controls.

Per-`.npz` fields (all length `n_res` or `(n_res, 24, ...)`; the first index
axis is **residues**, not atoms — `to_atomX` reduces per-atom quantities
that are constant within a residue back to `(n_res,)`, `convert.py:299-302`):

| field | shape | dtype | meaning |
|---|---|---|---|
| `residue_type` | `(n_res,)` | `<U…` str | one of `AA, DNA, RNA, METAL, SMOL, HOH` |
| `residue_name` | `(n_res,)` | str | 3-letter (AA) or 1-2 letter (NA) residue code |
| `residue_index` | `(n_res,)` | int (platform default via bare `np.array(list_of_int)`, effectively int64) | `label_seq` if present, else `seqid.num`, else `-1` |
| `chain_index` | `(n_res,)` | str | raw mmCIF/PDB chain letter (`chain.name`) |
| `entity_index` | `(n_res,)` | int32 | mmCIF `entity_id`, or `0` if absent |
| `molecule_index` | `(n_res,)` | int | unique id per covalently-linked entity (subchain), `HOH` waters each get a distinct synthetic id (`convert.py:256-261`) |
| `atom_mask` | `(n_res, 24)` | bool | atom resolved in this structure |
| `position` | `(n_res, 24, 3)` | float | atom coordinates (0 where unresolved) |
| `atom_name` | `(n_res, 24)` | str | e.g. `"CA"`, `""` for pad slots |
| `atom_type` | `(n_res, 24)` | str | element symbol, `""` for pad slots |
| (bond fields, atom-order index, bfactor) | — | — | **not read by `ProteinPDB`**; used only by `AtomPDB` (unused for `default_vp`, §A.5) |

### A.1.2 The directory layout an `AllPDB`-family dataset expects

From `compute_index` (`allpdb.py:1429-1461`, assembly-mode — this is what
`AllPDB(..., assembly=True)`, the default, uses):
```
{path}/entries.idx                              # PDB entries.idx: tab-separated
                                                 #   code \t ... \t date(DD/MM/YY) \t ... \t resolution \t exptype
{path}/{seqres_aa}_cluster.tsv                  # mmseqs2 cluster file, default name "clusterSeqresAA"
{path}/{seqres_na}_cluster.tsv                  # same, for nucleic acids ("clusterSeqresNA")
{path}/chain_assemblies.csv                     # rows: pdbid,chain,assembly_npz_filename
{path}/assembly{npz_version}_npz/{2-char}/{name}.npz   # per-structure caches, sharded into
                                                 #   2-character subfolders (chars [1:3] of the
                                                 #   uppercased 4-letter PDB code + assembly suffix)
```
`npz_version` defaults to `""`; `ProteinPDB`/`BatchedProteinPDBStream` never
override it (so `assembly_npz/`, no suffix) except `AtomPDBStream` and
`ProteinSMOLNeighboursPDB` which use `"_v3"` (`allpdb.py:285, 410`, both
unused for `default_vp`).

**mmseqs2 cluster file format** (`read_clusters`, `allpdb.py:1325-1348`):
tab-separated `representative_member \t member`, one line per member, grouped
by representative into clusters. `create_cluster_weights` (used only by
`AllPDBSample`, not by the `default_vp` path) assigns each member weight
`1/cluster_size`.

**`entries.idx` filter** (`entry_filter` closure inside `compute_index`,
`:1433-1437`): a residue's parent PDB entry is included only if
`resolution <= cutoff_resolution` (`4.0` for `default_vp`) **and**
`start_date <= deposition_date < cutoff_date` (defaults `"01/01/90"` to
`"12/31/21"` — `default_vp`'s training script does not override these, so
**the training cutoff is entries deposited before 2021-12-31**).

## A.2 Index construction (`compute_index`, `create_chain_index`) — cheap, port as-is

`AllPDB.__init__` calls `compute_index(path, start_date, cutoff_date,
cutoff_resolution, seqres_aa, seqres_na, npz_version)` **once**, at dataset
construction (`allpdb.py:1429-1461`):
1. List every `.npz` file under `assembly{npz_version}_npz/`, group by
   uppercased 4-letter PDB code (`name.split("-")[0].upper()`) into
   `npz_assemblies: dict[code -> list[path]]`.
2. Parse `entries.idx` → `dict[code -> {date, resolution, exptype}]`.
3. Parse both cluster `.tsv` files → `clusters: dict[representative -> set[member]]` for AA and NA separately (`read_clusters`).
4. `create_chain_index(clusters, entries, npz_assemblies, chain_assemblies, entry_filter)` (`:1384-1416`) builds, **per cluster**, a filtered list of
   `dict(entry=pdb_code, chain=chain_letter, assemblies=[...], metadata=...)`
   for every cluster member that (a) has a known, filter-passing PDB entry and
   (b) has at least one available assembly file for that specific chain (via
   `chain_assemblies.csv`, or falls back to "any assembly for this entry" if
   `chain_assemblies=None`, e.g. the asymmetric-unit variant `compute_asym_index`). Empty clusters (no member survives filtering) are dropped.
5. Result: `dict(AA=aa_cluster_list, NA=na_cluster_list)`, each a **list of
   lists** — outer index = cluster, inner index = member within that cluster.

`ProteinPDB.__len__` returns `len(self.data_index["AA"])` — **the number of
surviving AA sequence clusters**, not the number of PDB entries or chains.
`AllPDB.__getitem__(kind, index)` (`:79-109`) treats `index` as a **cluster
index**: it picks a **uniformly random member of that cluster**
(`random.choice(data_index[index])`) and a **uniformly random biological
assembly** of that member (`random.choice(chosen_chain["assemblies"])`), then
loads that one `.npz` file and filters residues to
`filter_residue_type` (`["AA"]` for `ProteinPDB`).

**This is the AlphaFold-style cluster-reweighting scheme**: because indexing
is over clusters and each access re-rolls both the member and the assembly,
a highly redundant sequence cluster (many near-identical PDB entries)
contributes the **same total sampling weight** as a singleton cluster over
one epoch, and every access to the same cluster index can return a
*different* structure. "One epoch" = one full pass over cluster indices,
implemented by `BatchedProteinPDBStream.get_next_pdb` shuffling
`list(range(len(self.protein_pdb)))` and popping until exhausted, then
reshuffling (prints `"shuffling..."`, `allpdb.py:847-861`).

## A.3 One `ProteinPDB` example, key by key (`allpdb.py:144-192`)

`ProteinPDB(AllPDB)` is constructed with `filter_residue_type=["AA"]`
(forced in `__init__`, ignoring any user override — `:154-157`). Its
`__getitem__(index)`:
```python
raw_data, chain = super().__getitem__("AA", index)   # AllPDB.__getitem__, loads+filters one .npz
aa_gt = argmax(raw_data["residue_name"][:,None] == AA_ORDER[None,:], -1)   # AA_ORDER: 21-entry alphabetical list, "UNK" last
aa_gt = where(no AA_ORDER match, 20, aa_gt)           # 20 = UNK/unresolved sentinel (matches "unk_restype_index" elsewhere)
```
`AA_ORDER` (`allpdb.py:30-33`): `['ALA','ARG','ASN','ASP','CYS','GLN','GLU',
'GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR',
'VAL','UNK']` — **identical order** to `restypes` used in
`02_schedules_noising_sampling_pdb.md` §4.3 with `'UNK'` appended as index 20
(the two lists must be kept index-compatible: `restypes[i] == AA_ORDER[i]`
for `i in 0..19`).

Returned dict (all `(n_res,)` except positions/masks), **directly from the
raw `.npz` fields, no atom14 truncation yet, no centering yet**:

| key | shape | dtype | source |
|---|---|---|---|
| `aa_gt` | `(n_res,)` | int (argmax output, effectively int64) | derived, 0-20 |
| `residue_index` | `(n_res,)` | int | `.npz` `residue_index`, as-is |
| `chain_index` | `(n_res,)` | **str** | `.npz` `chain_index`, **not yet numericalized** |
| `entity_index` | `(n_res,)` | int32 | `.npz`, as-is — **never read by `StructureDiffusion`; dead weight carried through the pipeline for `default_vp`** |
| `all_atom_positions` | `(n_res, 24, 3)` | float | `.npz` `position`, **atom24, not atom14** |
| `all_atom_mask` | `(n_res, 24)` | bool | `.npz` `atom_mask`, atom24 |

Plus the returned `chain` (a single Python string, the chain letter this
draw is "centered on" — used downstream to select which chain is the
"selected chain" for length filtering, §A.4).

**Atom14 truncation (`pos[:, :14]`) does NOT happen at the `Dataset` level.**
It happens once, inside the (JAX) training step, in
`StructureDiffusion.prepare_data` (`noise_schedule_benchmark.py:106-160`,
already specified as the loss's data-flow entry point in
`03_training_losses.md` §1.1 — do not re-derive it here, just note: a torch
port may truncate at collate time for a small memory win, but must not
reorder or reinterpret atoms when doing so — slice `[..., :14, :]` verbatim).

## A.4 `BatchedProteinPDBStream` — the actual `default_vp` datasource (`allpdb.py:812-991`)

Constructed by the training script as (`train_noise_schedule_benchmark.py:160-167`):
```python
BatchedProteinPDBStream(f"{opt.data_path}/allpdb/",
    seqres_aa="clusterSeqresAA", cutoff_resolution=4.0,
    p_complex=opt.p_complex,       # default 0.5
    size=opt.num_aa,               # default 1024 — residues per draw
    min_size=16, max_size=opt.num_aa,
    legacy_repetitive_chains=(opt.legacy_repetitive == "True"))   # default True
```
This is a `torch.utils.data.IterableDataset`; `__iter__` yields
`self.next_item()` forever (`:989-991`). `next_item` (`:863-987`) is a
**bin-packing loop**: repeatedly draw a random cluster/member/assembly via
`get_next_pdb` (§A.2), and either add the whole thing, add just one chain,
or reject it, until the running residue count reaches `size` (`opt.num_aa`).
Exact algorithm, in order, per drawn structure:

1. **Drop fully-unresolved-backbone residues immediately**: `slice_dict(protein_data, all_atom_mask[:, :3].all(-1))` (`:882`) — a residue survives only if **N, CA, C (atom24 slots 0,1,2) are all resolved**. Comment at `:874` flags this as a `FIXME` (the intent was apparently "drop unless all backbone atoms present", but this only checks 3 of 4 canonical backbone atoms — O is not required here; reproduce as written).
2. **Reject if `item_size < min_size`** (16) — `item_size` = residues remaining in the whole structure/complex after step 1, across all chains.
3. Identify `selected_chain = (chain_index == chain)` (the specific chain `AllPDB` centered this draw on) and its size. **Reject** (skip to next draw, no partial credit) if `selected_chain_size < min_size` or `> max_size` (both `16`/`1024` for `default_vp`, i.e. effectively just `< 16` since `max_size == size`), or if `selected_chain_size > size` (redundant with the previous check when `max_size==size`), or if `selected_chain_size == 0`.
4. **Repetitive-sequence filter** (`:901-908`): compute per-amino-acid counts either over `aa_gt` for **the whole structure** (`legacy_repetitive_chains=True`, `default_vp`'s actual setting) or over `aa_gt[selected_chain]` only (`False`). Reject if any single amino acid exceeds 50% of `selected_chain_size`. **Known upstream bug, reproduce as-is**: with `legacy_repetitive_chains=True` (the default and what `default_vp` used), the numerator counts amino acids **across the entire complex**, not just the selected chain, so in a hetero-complex a low-complexity *other* chain can wrongly veto an otherwise-fine selected chain, or vice versa — a comment at `:740-742` (in the sibling `BatchedProteinPDB` class) calls this out explicitly as a known complex-mode bug that was kept for reproducibility.
5. **AlphaFold-style length-based skip** (`:910-913`): `skip_chance = 1 - clamp(selected_chain_size, 256, 512) / 512`; reject with probability `skip_chance` (i.e. chains ≤256 residues are skipped with probability `0.5`, chains ≥512 are never skipped by this rule, linear in between). This is independent of and in addition to the repetitive/min-size filters.
6. **Requeue-or-reject on overflow** (`:918-926`): if adding the *selected chain alone* would already overshoot the remaining budget (`total + selected_chain_size > size`), push this structure's cluster index onto `requeue` (re-tried on the *next full epoch*, not immediately) and either `continue` (try another structure this same draw, if `total < size/2`) or `break` (finalize this draw as-is, if the batch is already reasonably full).
7. **Otherwise, decide monomer vs. complex** (`:927-941`): with probability `1 - p_complex` (`p_complex=0.5` ⇒ 50/50), take **just the selected chain**. Otherwise (complex mode): if the **whole structure** fits in the remaining budget (`total + item_size <= size`), take the **entire multi-chain complex**; if it doesn't fit, fall back to taking just the selected chain anyway.
8. `part["chain_index"] = numerical_chain_index(part["chain_index"])` (`:944` / `1235`): **within this one part only**, remap the (still-string) chain letters to small integers **in sorted-string order** (`np.unique` sorts; `numerical_chain_index`, `allpdb.py:1237-1240`) — ASCII sort, so e.g. uppercase letters sort before lowercase. This numbering is **local to this one accepted part**, not global across the whole eventual `size`-residue draw.
9. Append `part` to `result`, record `batch_index = count` (an incrementing integer, one value per **accepted part**, i.e. one value per monomer-or-complex accepted into this draw — **not** per original PDB structure and **not** per polymer chain within a complex: an entire accepted complex shares one `batch_index` value even though it may have several `chain_index` values). `total += part_size`; loop while `total < size`.
10. After the loop: `self.current_index = self.current_index + requeue` (deferred oversized items go back into the pool, appended — since `list.pop()` pops from the **end**, a `:976` comment flags this as possibly not doing what was intended: freshly-requeued items sit at the end and get popped again almost immediately, rather than being deferred to later in the epoch).
11. Concatenate all parts along axis 0 (`np.concatenate` per field, `:978-981`), attach the `batch_index` array, then **`pad_dict(result, self.size)`** (`:983`) zero-pads every field up to exactly `size` rows and adds `result["mask"]` (`True` for the first `total` rows, `False` for the padded tail — `pad_item`, `:1250-1271`).
12. `result["seq_mask"] = result["mask"] * (result["aa_gt"] != 20)`;
    `result["residue_mask"] = result["mask"] * result["all_atom_mask"].any(-1)`.

**Net contract: every `next_item()` call returns a dict of `size`-length
(`= opt.num_aa`, default 1024) flat arrays**, containing 0 or more accepted
monomers/complexes packed end-to-end, zero-padded at the tail, with local
per-draw `batch_index` (one integer per accepted part) and local per-part
`chain_index` (numeric, reset per part — **not** globally unique across
parts sharing the same draw; only `(chain_index, batch_index)` pairs are
unique, which is exactly why `StructureDiffusion.prepare_data` calls
`unique_chain(chain, batch)`, already ported, before using chain identity
for anything).

### Full per-example schema at the `BatchedProteinPDBStream` output boundary

| key | shape | dtype | notes |
|---|---|---|---|
| `aa_gt` | `(1024,)` | int | 0-20, `20` in pad region too (zero-fill) |
| `residue_index` | `(1024,)` | int | zero-filled in pad region — **not** a valid residue index there, gated by `mask` |
| `chain_index` | `(1024,)` | int32 | local per-part numeric id, zero-filled in pad region |
| `entity_index` | `(1024,)` | int32 | unused downstream |
| `all_atom_positions` | `(1024, 24, 3)` | float32 | zero-filled in pad region |
| `all_atom_mask` | `(1024, 24)` | bool | all-`False` in pad region |
| `batch_index` | `(1024,)` | int32 | one value per accepted part; **0 in the pad region too** (collides with real part 0 — harmless only because `mask=False` there, see §A.8/B1) |
| `mask` | `(1024,)` | bool | `True` for real rows, `False` for padding |
| `seq_mask` | `(1024,)` | bool | `mask & (aa_gt != 20)` |
| `residue_mask` | `(1024,)` | bool | `mask & all_atom_mask.any(-1)` |

This dict (or, for `p_complex`/size variants, the analogous `CroppedPDBStream`
output — see below) is exactly `item_0` in the training script
(`train_noise_schedule_benchmark.py:180-184`), and is exactly what
`StructureDiffusion.prepare_data`/`prepare_condition` consume (already
specified in `03_training_losses.md` §1.1-1.3).

### `CroppedPDBStream` (`allpdb.py:993-1116`) — an alternate cropping policy, not used by `default_vp`'s training script but worth knowing

Subclasses `BatchedProteinPDBStream` and overrides `next_item`/adds `crop`:
instead of bin-packing multiple structures to fill `size`, it takes **one**
structure per draw and, if it's larger than `size`, **crops** it — either a
contiguous random window (single chain), a spatial crop around a random
interface contact residue (complex, `p_spatial=0.5`), or a per-chain
budgeted contiguous crop that walks chains in random order allocating
`randint(min(16,chain_length,budget), min(chain_length,budget))` residues
each (`:1041-1116`). Not referenced by `train_noise_schedule_benchmark.py`'s
`opt.dataset` branches — **dead for `default_vp`**; documented here because
the writer may need it for other configs and it shares infrastructure with
§A.4.

## A.5 Other dataset classes in `allpdb.py`/`afdb.py`/`synthete.py`/`mix_batch.py` — NOT used for `default_vp`

All of the following exist in the same files but are unreachable from
`train_noise_schedule_benchmark.py`'s `default_vp` invocation. Port only if
targeting a different config:

- **`afdb.py` (`AFDB`, `AFDBStream`, `CroppedAFDBStream`)**: module docstring
  says **"Not used in the manuscript."** Reachable from the training script
  only via `--dataset afdb*`, which `default_vp`'s canonical run did not use
  (dataset defaults to `"PDB"`). Loads single-model AlphaFoldDB predictions
  (`AF-{name}-F1-model_v4.npz`, `all_chains == 0` since AFDB has one chain),
  derives per-atom pLDDT from a `bfactor` field, and supports length-balanced
  resampling (`length_weights`, inverse-frequency over 100-residue-wide
  bins) to counter AFDB's bias toward 100-300 residue proteins.
- **`synthete.py` (`Synthete`, `SyntheteStream`)**: a fixed-256-residue
  synthetic dataset with precomputed ESM self-consistency filtering
  (`sc_rmsd <= 2.0`, `plddt > 70`), reachable via `--dataset SYNTHETE`
  (forces `num_aa=256`, `train_noise_schedule_benchmark.py:132-133`). Not
  used by `default_vp`.
- **`mix_batch.py` (`MixBatch`)**: a generic weighted-random-choice wrapper
  over several `IterableDataset`s (`random.choices(iters, weights=...)`).
  **Never imported by `train_noise_schedule_benchmark.py`** — available
  infrastructure, unused for `default_vp`.
- **`allpdb.py`'s `AtomPDB`/`AtomPDBStream`, `ProteinSMOLNeighboursPDB`,
  `ProteinBinderPDB`/`BinderPDBStream`, `ProteinAtomPDB` (marked
  `DEPRECATED`), `ProteinCropPDB`, `AllPDBSample`**: support all-atom/ligand
  (LigandMPNN-style), binder-design, and fixed-crop training variants for
  *other* salad configs; none are reachable from `default_vp`'s training
  script. `periodic_table.py`/`elements.py` (module docstring: **"Not used in
  the manuscript"**) back only `AtomPDB`'s periodic-table featurization —
  dead for `default_vp`.

## A.6 `salad/modules/utils/dssp.py` — `assign_dssp`/`drop_dssp` (full spec)

Called once per training step, inside the (JAX-)jitted forward pass, from
`prepare_condition` (`noise_schedule_benchmark.py:234-239`) — **not** a
`Dataset`-level / CPU-numpy preprocessing step in the JAX original (it runs
on-device, batched, every step, using the already-centered `atom_pos` atom14
ground truth). Only its first return value, `dssp` (3-state per-residue
secondary structure, `0=loop,1=helix,2=strand`), is consumed downstream
(§0c). Port `assign_dssp` fully; `drop_dssp` only if you want RNG-stream
parity (its output is otherwise unused).

### A.6.1 `get_hbond_map(coord, mask, cutoff=-0.5, margin=1.0)` (`dssp.py:60-94`)

Adapted from PyDSSP. `coord` is `(L, ≥4, 3)` (only slots 0-3 = N,CA,C,O are
read, `:71`); `mask` is a **pair mask** `(L,L)` bool (same-batch-item AND
both-residues-valid), passed in by the caller, not computed here.
```text
h = _get_peptide_bond_h_position(coord)      # (L-1, 3): backbone amide H position for residues 1..L-1, geometrically
                                              #   placed from unit(C[i]->N[i+1]) + unit(CA[i+1]->N[i+1]), normalized, *1.01 A from N
nmap = coord[1:, None, N]     ; hmap = h[:, None]                # both (L-1, 1, 3)
cmap = coord[None, :-1, C]    ; omap = coord[None, :-1, O]       # both (1, L-1, 3)
d_on, d_ch, d_oh, d_cn = pairwise euclidean distances of (O_i,N_j)/(C_i,H_j)/(O_i,H_j)/(C_i,N_j), each +1e-3
e = 0.084 * (1/d_on + 1/d_ch - 1/d_oh - 1/d_cn) * 332            # Kabsch-Sander electrostatic H-bond energy, (L-1, L-1)
e = pad(e, [[1,0],[0,1]])                                        # -> (L, L), row 0 and last col are zero-padding (no H at residue 0, no C/O acceptor at residue L-1)
local_mask = NOT(eye(L)) AND NOT(diag(k=-1)) AND NOT(diag(k=-2))  # exclude (i,i),(i,i-1),(i,i-2) — i.e. i as DONOR excludes itself and its 2 preceding residues as acceptor
hbond_map = clip(cutoff - margin - e, -margin, margin)
hbond_map = (sin(hbond_map / margin * pi/2) + 1) / 2              # smooth sigmoid-like relaxation of the hard "e < cutoff" DSSP criterion, in [0,1]
hbond_map = where(local_mask, hbond_map, 0)
hbond_map = where(mask, hbond_map, 0)                             # caller's pair_mask
return hbond_map                                                  # (L, L), hbond_map[i,j] ~ strength of H-bond where j is DONOR (see transpose below), i is ACCEPTOR? see A.6.2 for the actual index convention used
```
Note the `k=-1`/`k=-2` local mask excludes only residues **before** `i` in
the exclusion set as written (`jnp.diag(..., k=-1)` zeros the sub-diagonal,
i.e. `(i, i-1)`) — combined with the caller swapping axes immediately after
(`assign_dssp:115`), the net local-exclusion window ends up symmetric around
each residue; port `_unfold`/`get_hbond_map`/`assign_dssp` together rather
than trying to re-derive the index convention from this function alone — the
exact row/col meaning only needs to be self-consistent, and it is, by
construction (the two functions are always used as a pair, never separately).

### A.6.2 `assign_dssp(coord, batch_index, mask)` (`dssp.py:96-166`)

`coord` is atom14 (or wider) `(L, ≥14, 3)` **ground-truth** positions (called
on `data["atom_pos"]`, already atom14, already centered);
`mask` here is **residue validity** (`pos_mask.any(-1)` at the call site,
i.e. "this residue has at least one resolved atom"), **not** a pair mask —
`assign_dssp` builds its own pair mask internally.

**Degenerate short-input guard** (`:99-107`): if `coord.shape[0] < 5`
(fewer than 5 residues in the **whole flattened multi-item batch**, not per
item — an easy trap), skip all computation and return `dssp = 3` (an
**out-of-range sentinel**, not one of `{0,1,2}`!) for every residue, empty
blocks, and an all-`False` adjacency. **This 3-valued sentinel is exactly
the same value used elsewhere as "no DSSP conditioning provided"**
(`noise_schedule_benchmark.py:492`: `dssp = 3 * ones_like(...)` as the
no-condition default) — reproduce it exactly; do not clamp it into `{0,1,2}`.
In practice this guard only fires for pathologically tiny batches (test
fixtures), never real `size=1024` training draws.

Main path:
```text
pair_mask = (batch_index[:,None]==batch_index[None,:]) * mask[:,None] * mask[None,:]   # (L,L)
hbmap = get_hbond_map(coord, pair_mask)          # (L,L)
hbmap = swapaxes(hbmap, -1, -2)                  # transpose — see A.6.1 note
turn3 = diagonal(hbmap, offset=3) > 0            # (L-3,) bool: hbmap[i, i+3] > 0
turn4 = diagonal(hbmap, offset=4) > 0            # (L-4,)
turn5 = diagonal(hbmap, offset=5) > 0            # (L-5,)
h3 = pad(turn3[:-1] * turn3[1:], [1,3])          # (L,) two consecutive turn3's -> a helix-3 nucleus, right-padded with 3 zeros (+1 left) to length L
h4 = pad(turn4[:-1] * turn4[1:], [1,4])
h5 = pad(turn5[:-1] * turn5[1:], [1,5])
helix4 = h4 + roll(h4,1) + roll(h4,2) + roll(h4,3)      # OR of h4 nucleus at this residue or the preceding 3 (an alpha-helix "core" spans 4 residues from one nucleus)
h3 = h3 * NOT(roll(helix4,-1)) * NOT(helix4)            # helix4 takes priority over helix3 at any residue helix4 already claims (this OR the next)
h5 = h5 * NOT(roll(helix4,-1)) * NOT(helix4)            # same priority rule for helix5
helix3 = h3 + roll(h3,1) + roll(h3,2)
helix5 = h5 + roll(h5,1) + roll(h5,2) + roll(h5,3) + roll(h5,4)
# --- bridge / strand detection ---
unfoldmap = (_unfold(_unfold(hbmap, 3, axis=-2), 3, axis=-2)) > 0    # (L-2, L-2, 3, 3) sliding 3x3 windows of hbmap along BOTH axes
unfoldmap_rev = swapaxes(unfoldmap, 0, 1)
p_bridge = unfoldmap[:,:,0,1]*unfoldmap_rev[:,:,1,2] + unfoldmap_rev[:,:,0,1]*unfoldmap[:,:,1,2]   # "parallel bridge" pattern
p_bridge = pad(p_bridge, [[1,1],[1,1]])           # back to (L,L)
p_bridge = where(pair_mask, p_bridge, 0)
a_bridge = unfoldmap[:,:,1,1]*unfoldmap_rev[:,:,1,1] + unfoldmap[:,:,0,2]*unfoldmap_rev[:,:,0,2]   # "antiparallel bridge" pattern
a_bridge = pad(a_bridge, [[1,1],[1,1]]); a_bridge = where(pair_mask, a_bridge, 0)
ladder = (p_bridge + a_bridge).sum(-1) > 0        # (L,) this residue participates in >=1 bridge of either kind, to ANY partner
p_res = p_bridge.sum(-1) > 0 ; a_res = a_bridge.sum(-1) > 0
p_strand = (p_res * roll(p_res,-1) + p_res * roll(p_res,1)) > 0
a_strand = (a_res * roll(a_res,-1) + a_res * roll(a_res,1)) > 0
strand_unused = a_strand + p_strand + a_res + p_res > 0   # computed but IMMEDIATELY OVERWRITTEN below — dead intermediate, see note
helix = (helix3 + helix4 + helix5) > 0
strand = ladder            # <-- overwrites the `strand` computed just above from p_strand/a_strand/etc; those are dead
loop = NOT(helix) * NOT(strand)
index = argmax(stack([loop, helix, strand], -1), -1)      # (L,) int in {0,1,2}; ties broken by argmax's first-True-wins order: loop(0) > helix(1) > strand(2) priority
```
**Dead-code note**: `p_strand`/`a_strand` (and the `strand = a_strand +
p_strand + a_res + p_res > 0` line, `:147`) are computed and then
**immediately shadowed** by `strand = ladder` on the very next line (`:150`)
before `strand` is used anywhere. Do not port `p_strand`/`a_strand`/that
first `strand` assignment as anything other than dead computation you may
skip — the final secondary-structure `index` depends only on `helix` (from
`helix3/4/5`) and `ladder`.

**Block labeling via `lax.scan`** (`:153-158`), the "CPU-preprocessing,
sequential-dependency" piece flagged in the task brief:
```python
def block_iter(carry, data):
    state, block_index = carry
    new_state = data
    block_index = where(new_state != state, block_index + 1, block_index)
    return (new_state, block_index), block_index
_, blocks = jax.lax.scan(block_iter, (index[0], 0), index)
```
This is a strictly sequential O(L) recurrence (each step's carry depends on
the previous), which is exactly the shape of computation `lax.scan` exists
for and which has **no efficient parallel primitive in plain PyTorch** — a
literal port would be a slow per-residue Python loop. **It is unnecessary:
this recurrence is exactly a run-length "block id" counter, equivalent to a
prefix-sum of a boolean "value changed from previous element" array.**
Vectorized torch reformulation (bit-exact):
```python
changed = torch.zeros_like(index, dtype=torch.bool)
changed[1:] = index[1:] != index[:-1]         # changed[0] is always False (matches scan's init carry == index[0])
blocks = torch.cumsum(changed.long(), dim=0)  # exactly reproduces the scan's `blocks` output
```
**Caveat carried over from §0(c): as noted, `blocks` (and everything built
from it — `same_block`, `block_distance`, `block_adjacency`) is dead for
`default_vp`'s actual loss/conditioning.** It is specified here in full only
because `assign_dssp` cannot be called with just the `dssp` output without
either computing or explicitly skipping this part; the recommended torch
port computes `blocks` via the cumsum trick above (cheap, correct) but the
writer may skip `block_adjacency`'s scatter-min entirely (next paragraph) if
only `dssp` is needed, by simply not implementing that return value in a
lean `assign_dssp` variant, as long as the eventual API still returns a
same-shaped `(L,)` and `(L,L)` bool array of the right dtype for any caller
expecting the full 3-tuple (fill with zeros/`False`, clearly documented as
"unused by default_vp").

**`block_adjacency`** (`:159-164`), specified for completeness /
non-`default_vp` configs:
```text
same_block = blocks[:,None] == blocks[None,:]                       # (L,L)
sheet_sheet = (index[:,None]==2) * (index[None,:]==2)                # both residues are STRAND
distance = ||coord[:,None,CA] - coord[None,:,CA]||                   # (L,L) plain CA-CA distance, NO pair_mask applied here
# scatter-MIN (not scatter-add): for every (i,j), find the min CA-CA distance among
# ALL residue pairs (i',j') that map to the same (block[i'], block[j']) cell, using an
# (L,L)-sized accumulator (over-allocated; only the first num_blocks x num_blocks
# cells are ever meaningfully read back, but indices stay in-bounds since num_blocks <= L)
block_distance = scatter_min(init=1e6 * ones((L,L)), index=(blocks[:,None], blocks[None,:]), src=distance)[blocks[:,None], blocks[None,:]]
block_adjacency = where(sheet_sheet, block_distance <= 8.0, block_distance <= 8.0)   # SHEET_ADJACENCY == OTHER_ADJACENCY == 8.0 -- both branches numerically identical for default_vp's constants; the sheet_sheet split exists so a config could set them differently, it just never does
block_adjacency = where(same_block | loop[:,None] | loop[None,:], 0, block_adjacency)  # zero out same-block pairs and any pair touching a LOOP residue
```
`scatter_min` in torch: build a dense `(num_blocks, num_blocks)` accumulator
(cheaper than replicating the `(L,L)`-sized over-allocation) initialized to
`1e6`, and use `Tensor.scatter_reduce_(dim, index, src, reduce="amin")` (or
loop-free `index_put_` is not order-safe for min — use `scatter_reduce_`),
then gather back via `block_distance_full[blocks][:, blocks]`. **Correctness
note, not a bug**: the block-labeling recurrence above does **not** reset at
`batch_index`/chain boundaries, so a block can in principle span the
tail of one packed protein and the head of the next one in the same flat
draw if their terminal/initial `index` (loop/helix/strand) values happen to
match. This never corrupts `dssp` itself (every upstream quantity feeding
`index` — `hbmap`, `p_bridge`, `a_bridge` — is already `pair_mask`-gated to
same-batch-item pairs only), and any resulting cross-item block gets
`same_block=True` for the boundary residue pair, which unconditionally
zeroes `block_adjacency` there anyway — so the (unused, for `default_vp`)
`block_adjacency` output happens to be safe by downstream construction, not
because the block-labeling itself is batch-aware. Flag this precisely if you
ever revive `block_adjacency` for another config; do not silently "fix" the
scan to reset at boundaries without checking whether that changes the
(currently harmless) numeric result some other config relies on.

### A.6.3 `drop_dssp(key, secondary_structure, blocks, block_adjacency, p_drop=0.2)` (`dssp.py:168-173`)

```text
drop_mask = bernoulli(key, p_drop, shape=secondary_structure.shape)[blocks]   # per-BLOCK bernoulli draw, broadcast to every residue in that block via `blocks` indexing
drop_pair = drop_mask[:,None] | drop_mask[None,:]
secondary_structure = where(drop_mask, 0, secondary_structure)     # dropped residues get relabeled to class 0 (LOOP) -- not a separate "unknown" sentinel
block_adjacency = where(drop_pair, 0, block_adjacency)
return secondary_structure, block_adjacency
```
`p_drop_dssp = 0.2` for `default_vp`. **Dead for `default_vp`** per §0(c) —
its `secondary_structure` return is discarded, its `block_adjacency` return
is unused downstream. Implement only if you need RNG-stream parity or plan
to use it for another config.

## A.7 `salad/modules/utils/augmentation.py` — dead for `default_vp`

`random_rotation(key)`, `augment_rotation(key, pos)`, `augment_index(key,
index, size=1000)`, `augment_category(key, index, num_classes=1000)`
(`augmentation.py:1-27`). **Grepped: the only importer anywhere in the
repository is `salad/modules/experimental/old_sequence_design.py`** —
not reachable from `default_vp`'s training script. Do not port unless
targeting that experimental config.

The augmentation actually *wired into* `default_vp`'s model code is a
**different, similarly-named** set of functions living in
`salad/modules/structure_autoencoder.py:1081-1119`
(`structure_augmentation`, `structure_augmentation_params`,
`apply_structure_augmentation`, `apply_inverse_structure_augmentation`,
imported by `noise_schedule_benchmark.py:34-38`) — and per §0(a), **these
too are dead for `default_vp`** because every call site is gated on
`c.equivariance == "semi_equivariant"`, which `default_vp` never sets. For
completeness (needed if you port a `semi_equivariant_*` config later):
```text
structure_augmentation_params(pos, batch, mask):
    center = index_mean(pos[:,1], batch, mask[:,None])              # per-batch-item CA centroid
    translation = normal(key, pos[:,1].shape)[batch]                 # per-batch-item random translation, broadcast to residues
    rotation = random_rotation(batch)                                 # per-RESIDUE(!) random rotation matrix -- despite the name suggesting per-batch-item, `random_rotation` here (structure_autoencoder.py's own version, NOT augmentation.py's) is called with `batch` as its "key" argument in a way that produces one rotation per residue index, not one per batch item -- if you ever port this, verify against structure_autoencoder.py directly rather than assuming batch-level sharing
    return center, translation, rotation
apply_structure_augmentation(pos, center, translation, rotation):
    pos = pos - center[:,None]
    pos = rotation[:,None].apply_to_point(pos)                        # Vec3Array rotation application
    pos = pos + translation[:,None]
    return pos
apply_inverse_structure_augmentation: exact inverse, translation undone first, then rotation.inverse(), then + center
```
Not required for `default_vp` — included only so the writer implements it
correctly, gated by the same config flag, if another config needs it later.

## A.8 What the torch `Dataset`/`DataLoader` must produce

**Dataset**: a `torch.utils.data.IterableDataset` mirroring
`BatchedProteinPDBStream.next_item()` (§A.4) — each `__iter__` step yields
one dict of `size`-length (`opt.num_aa`, e.g. 1024) flat numpy/torch arrays,
zero-padded at the tail, with the exact key/shape/dtype table in §A.4,
built by the bin-packing algorithm specified there (steps 1-12), including
its documented quirks (the `legacy_repetitive_chains` complex-mode bug, the
`min(16,...)`/backbone-only drop check, the requeue-pops-from-the-end
ordering wrinkle) — reproduce them unless explicitly asked to fix them, and
if you do fix any, log the deviation clearly since it changes what the model
sees during training.

**Collation for gradient accumulation / multi-worker throughput — MUST
concatenate, never pad-and-stack.** The upstream `DataLoader` wrapper,
`flexloop.data.BatchStream` (`flexloop/data.py:91-116`), constructs:
```python
DataLoader(dataset, batch_size=accumulate, shuffle=False,
           collate_fn=np_collate, num_workers=32, prefetch_factor=32,
           persistent_workers=True, drop_last=False, timeout=120,
           worker_init_fn=_worker_init_function)
```
where `accumulate` here is the training script's
`rebatch * accumulate * NUM_DEVICES` (§0f), and
```python
def np_collate(items):
    return {key: np.concatenate([i[key] for i in items], axis=0)
            for key in items[0].keys() if all(key in b for b in items)}
```
**`np.concatenate`, not `np.stack`.** A batch of `K` dataset draws (each
already a fixed-`size`-length flat array with local `batch_index`) becomes
one `(K*size,)`-length flat array — never a `(K, size, ...)` tensor. This is
mandatory, not a style choice: `salad`'s segment ops (`index_mean`,
`index_sum`, `extract_neighbours`, `get_neighbours`, ...) all expect a flat
`(N,)`-indexed input with a companion `batch_index`/`mask`; introducing a
real leading batch dimension would be architecturally incompatible with
every downstream op in `salad_torch/geometry/segment.py`. **After
concatenating for throughput, the consumer must split the flat `(K*size,)`
array back into `K` separate `size`-length chunks (by reshape) before
running any forward pass** — reproducing what `hk.vmap`/`jax.lax.scan`
do upstream (§0f) — and process each chunk with **its own local
`batch_index`**, never merging chunks' `batch_index` spaces. `torch`'s
default `DataLoader` collate (`torch.stack`, implicit zero-dim batching) is
the wrong tool here; either use a custom `collate_fn` doing exactly the
`np.concatenate` above, or (simplest for a single-GPU port, since
`rebatch=accumulate=NUM_DEVICES=1` by default) skip a `DataLoader` batch
dimension entirely and treat each dataset draw as one training step.

`worker_init_fn=_worker_init_function` (`flexloop/data.py:138-143`) reseeds
Python's `random` and `numpy.random` per worker from
`torch.initial_seed() + worker_id`. **Because `BatchedProteinPDBStream` is
an `IterableDataset` with `split_across_workers=False` (the default, and
`default_vp`'s training script never overrides it), every one of the 32
DataLoader workers independently iterates the *entire* cluster index with
its own random shuffle order — `num_workers=32` here is purely a
prefetch-throughput parallelism trick (32 independent producers feeding one
consumer queue), not a proper dataset-sharding scheme.** There is no
guarantee against two workers emitting overlapping/duplicate draws in the
same wall-clock window; this is fine for an infinite streaming dataset with
random cluster reweighting (§A.2) and does not need to be "fixed" in a torch
port — just don't assume `num_workers` shards the cluster index the way a
map-style dataset with a `DistributedSampler` would.

**Padding is not safe — cross-reference `notes/04_PARITY_AND_PERF.md` §B1,
and it is NOT only an inference-time issue.** B1 documents that
`get_neighbours`'s stage-chained `+inf`-masking has a `-1`-sentinel bug: a
`-1` "no more neighbours" entry indexes column `N-1` with that residue's
*real, finite* distance, undoing the `+inf` a genuine neighbour at index
`N-1` had just received, causing residue `N-1` to be double-selected and the
answer for real residues to depend on batch/padding layout. **The exact same
`get_neighbours` call is used by the `local` loss (`03_training_losses.md`
§7.8, weight `10.0` — the single largest loss weight in the whole objective)
and `get_random_neighbours` (used by `fape`/`distogram`, §7.7/§7.9) run
every training step on a flat array that, per §A.4, is essentially always
padded** (bin-packing rarely fills `size` exactly, and `get_neighbours`
produces `-1` sentinels for any residue with fewer than `K`
(16 or 64) valid same-item, same-batch neighbours — common for the
`min_size=16` short chains this pipeline explicitly allows, and for residues
near the tail of any part that doesn't perfectly fill its own local
neighbourhood). **This means B1 is not a corner case reachable only via
unusual multi-batch inference calls — it is live in essentially every
`default_vp` training step**, contaminating the (heavily-weighted) `local`
loss to some degree via padding-dependent neighbour selection. This is an
upstream-faithful bug (verified bit-identical to JAX per B1's own testing)
— reproduce it for parity, but the writer should be aware it exists and
should not be surprised or assume a fresh implementation bug if a training
run's `local` loss curve doesn't exactly match a hypothetical
"canonically correct" neighbour-selection re-implementation. If a
correctness fix is ever wanted instead of parity, the right fix is inside
`get_neighbours` itself (already covered in `04_PARITY_AND_PERF.md` B1), not
in the data pipeline — do not try to "fix" this by trimming padding to
`total` residues per draw instead of the fixed `size` buffer, since that
changes the training distribution (variable-length flat arrays would need
every downstream shape-dependent op re-verified) for no real gain, given
B1's true fix is a one-line change inside neighbour selection.

---

# PART B: THE TRAINING LOOP

## B.1 `train_noise_schedule_benchmark.py` walkthrough (`salad/training/train_noise_schedule_benchmark.py`, 235 lines)

CLI options (`parse_options`, `flexloop/utils.py:1-26` — thin `argparse`
wrapper inferring `type=` from each default's Python type; **incidental**,
any `argparse`/config-dataclass equivalent is fine in torch) and their
values as `default_vp` actually ran (`:98-125`):

| option | default | meaning |
|---|---|---|
| `path` | `"network/"` | checkpoint/log root |
| `config` | `"default"` | **overridden per-run** to `"default_vp"` on the actual CLI invocation (the code default is a placeholder) |
| `data_path` | `""` | root for `{data_path}/allpdb/` etc. |
| `num_aa` | `1024` | residues per draw, `= size` passed to the dataset (§A.4); forced to `256` for `SYNTHETE`, `1024` for `afdb*` (`:132-135`) |
| `p_complex` | `0.5` | passed straight to `BatchedProteinPDBStream` |
| `lr` | `1e-3` | peak LR after warmup |
| `decay_lr` | `1e-5` | LR floor at the end of the cosine decay |
| `warmup_steps` | `1000` | linear warmup length |
| `decay_steps` | `500000` | cosine decay length (post-warmup) |
| `clip` | `0.1` | global-norm gradient clip threshold |
| `b1`, `b2` | `0.9`, `0.99` | Adam betas — **note `b2=0.99`, not the Adam-typical `0.999`** |
| `ema_weight` | `0.999` | EMA decay (§B.4) |
| `rebatch` | `1` | `hk.vmap` sub-batch count (§0f, cluster-scaling, ignore for single-GPU) |
| `accumulate` | `1` | gradient-accumulation micro-batch count (§0f) |
| `jax_seed` | `42` | top-level PRNG seed |
| `afdb_max_L`, `afdb_min_plddt` | `0.5`, `70.0` | unused unless `--dataset afdb*` (§A.5) |
| `multigpu` | `"True"` (string, compared literally) | enables `pmap` across all visible devices |
| `nanhunt` | `"False"` | debug: disables `multigpu`+`jit`, runs eager for NaN-hunting — **incidental debug tooling** |
| `dataset` | `"PDB"` | selects the `else` branch → `BatchedProteinPDBStream` (§0d) |
| `suffix` | `"1"` | checkpoint-path disambiguator, incidental |
| `legacy_repetitive` | `"True"` | → `legacy_repetitive_chains=True` (§A.4 step 4) |
| `schedule_fixed` | `"False"` | `"True"` swaps to the no-decay `warmup_schedule` (§B.3) |
| `order_agnostic` | `"False"` | only affects the (unused, §A.5) AFDB path |

Boolean-like options are **compared as literal strings** (`opt.multigpu ==
"True"`) because `parse_options` infers `argparse` types from the Python
`type()` of each default, and a `str` default stays a `str` — this is a
`flexloop`/CLI-parsing idiosyncrasy, not something to reproduce in a
dataclass-based torch config (just use real `bool`s there); noted so the
writer doesn't get confused reading `opt.multigpu == "True"` in the source.

Model/optimizer construction sequence (`:173-215`):
1. `config = getattr(config_choices, opt.config)` — looks up `default_vp` by
   name from `salad/modules/config/noise_schedule_benchmark.py`.
2. `key = jax.random.PRNGKey(opt.jax_seed)` (`42`).
3. `init, step = hk.transform(model_step(config, rebatch, is_training=True))`
   — Haiku's functional-params transform; **no torch equivalent needed**,
   `nn.Module` parameters are just attributes.
4. **One-time param-shape-inference forward pass**: `item_0 = next(data)`,
   then `init_batch = tree_map(lambda x: x[:rebatch*accumulate*100],
   item_0)` — **truncates the very first batch to only its first
   `rebatch*accumulate*100` residues** (`100` for `default_vp`'s
   `rebatch=accumulate=1`) purely to make the one-off `init(key,
   init_batch)` call cheap; Haiku allocates every parameter's shape lazily
   on first call, and none of this model's parameter shapes depend on the
   residue count, so a small probe batch is sufficient and faster to
   compile than a full `1024`-residue one. **This is Haiku-specific
   incidental setup — a torch `nn.Module`'s parameters exist as soon as
   `__init__` runs (assuming no lazy `nn.LazyLinear`-style modules are
   used), so there is nothing to port here beyond "construct the model
   once."**
5. `hk.experimental.tabulate(...)` writes a `model_description` text file —
   **Haiku introspection tooling, incidental**; `str(model)` /
   `torchinfo.summary` or nothing at all are fine substitutes.
6. Build `cosine_decay_schedule`/`warmup_schedule` (§B.3) and the `optax`
   optimizer chain (§B.2).
7. `loop_state = State(key, 0, params, opt_state, aux_state={})`; attempt
   `load_loop_state(path)` to resume (§B.5) — falls back to the freshly
   initialized state if no `save.jax` exists yet (`try/except`, prints
   `"Could not find save file, starting from random initialization..."`).
8. `training(path, make_training_inner(...), max_steps=warmup_steps +
   decay_steps + 1, logger=log())(writer, loop_state)` — runs forever (up to
   `max_steps = 501001` for the defaults) inside `simple_loop.training`
   (`flexloop/simple_loop.py:348-372`, §B.6).

## B.2 Optimizer (`train_noise_schedule_benchmark.py:202-210`)

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(opt.clip),          # 0.1
    optax.scale_by_adam(opt.b1, opt.b2, eps=1e-9), # 0.9, 0.99, eps hardcoded (not a CLI option)
    optax.scale_by_schedule(schedule),             # cosine_decay_schedule, see B.3
    optax.scale(-1.0))
opt_state = optimizer.init(params)
```
Order matters: **gradients are clipped by global L2 norm to `0.1` BEFORE
being fed into Adam's first/second-moment estimators.** This means Adam's
internal `m`/`v` state tracks the *clipped* gradient, not the raw one — a
subtlety that matters if a torch port used `torch.nn.utils.clip_grad_norm_`
*after* `optimizer.step()`'s internal moment update (it wouldn't, since that
function only mutates `.grad` in place and must be called before `.step()`
regardless — so the natural torch idiom already gets this order right):
```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
optimizer.step()   # torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), eps=1e-9), lr set per-step by the scheduler below
```
`optax`'s trailing `scale(-1.0)` and `optax.apply_updates` (`params =
params + updates`) is just `optax`'s "updates are added" convention versus
`torch.optim`'s "subtract the step" convention — **mathematically identical
to standard `torch.optim.Adam`**, no special handling needed. `eps=1e-9` is
inside the denominator exactly as PyTorch's `Adam.eps` is (`v_hat.sqrt() +
eps`, not `eps_root` inside the sqrt) — direct 1:1 correspondence, use
`torch.optim.Adam(..., eps=1e-9)`.

**Weight decay: none.** No `optax.add_decayed_weights`/`optax.adamw` term
anywhere in the chain — `default_vp` trains with plain Adam, zero weight
decay. Do not add `torch.optim.AdamW` or any `weight_decay > 0` — that would
be a deviation from upstream, not a neutral choice.

## B.3 Learning-rate schedule (`train_noise_schedule_benchmark.py:68-82`)

```python
def cosine_decay_schedule(start_lr, decay_lr, warmup_steps, decay_steps):
    def schedule(count):
        result = where(count <= warmup_steps,
                        count / warmup_steps * start_lr,                                  # linear warmup
                        (start_lr - decay_lr) * 0.5 * (1 + cos(pi * ((count - warmup_steps) % decay_steps) / decay_steps)) + decay_lr)
        return maximum(result, 0.0)
    return schedule
```
Values for `default_vp`: `start_lr=1e-3, decay_lr=1e-5, warmup_steps=1000,
decay_steps=500000`. **Linear warmup for `1000` steps** (`count/1000 *
1e-3`, reaching exactly `1e-3` at `count=1000` — note `count<=warmup_steps`
uses the warmup branch AT `count==warmup_steps` too, giving exactly
`start_lr`, then the next step, `count=1001`, evaluates the decay branch at
`(1001-1000)%500000=1` — continuous to within one step's resolution, no
discontinuity), then a **single half-cosine decay** from `1e-3` down to
`1e-5` over the next `500000` steps. **Latent quirk, reproduce exactly, not
a `default_vp`-relevant bug**: the `% decay_steps` modulo means that if
training ever continues **past** step `warmup_steps + decay_steps`
(`501000`), the schedule **restarts** the cosine cycle from `start_lr`
rather than holding at `decay_lr` — this is a warm-restart schedule with an
unbounded restart count baked in, but is inert for any run that stops at or
before the documented `max_steps = warmup_steps + decay_steps + 1`.
Reproduce the modulo faithfully (don't clamp it to "hold at `decay_lr`
forever") in case a resumed/extended run is expected to exhibit the restart.

**Alternate schedule** (`--schedule_fixed True`, `:78-82, 198-199`):
```python
def warmup_schedule(start_lr, warmup_steps):
    return lambda count: minimum(count / warmup_steps * start_lr, start_lr)   # linear warmup, then FLAT at start_lr forever, no decay
```
Not `default_vp`'s actual schedule but trivial to support as a config
option; port both.

`total_steps = opt.warmup_steps + opt.decay_steps + 1 = 501001` (`:197`) is
passed as `max_steps` to the training loop (§B.6).

Torch equivalent: implement as a plain function of the integer step count
(e.g. a `torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)`
where `schedule(step)` returns a **multiplier relative to `optimizer`'s
`lr=1.0`**, or simply set `param_group['lr'] = schedule(step)` directly each
step — the latter is more literal since `schedule` here already returns an
absolute LR, not a multiplier).

## B.4 EMA (`train_noise_schedule_benchmark.py:38-48`) — confirmed, decay `0.999`

```python
def ema_step(params, loop_state):
    if "ema_params" not in loop_state.aux_state:
        return dict(ema_params=deepcopy(params))                 # first call: EMA initialized to a copy of the CURRENT (pre-update) params
    ema_params = loop_state.aux_state["ema_params"]
    ema_params = tree_map(lambda x, y: ema_weight * x + (1 - ema_weight) * y, ema_params, params)
    return dict(ema_params=ema_params)
```
`ema_weight = opt.ema_weight = 0.999` for `default_vp` — a **standard
exponential moving average with decay 0.999** (i.e. an effective averaging
window on the order of `1/(1-0.999) = 1000` steps), the conventional choice
for diffusion-model EMA checkpoints. **Confirmed: `checkpoint-ema-{step}.jax`
files are what the manuscript's published eval/sampling results were
generated from** — `train_noise_schedule_benchmark.py:59-62` writes BOTH
`checkpointables = {"checkpoint": params, "checkpoint-ema":
aux_state["ema_params"]}` every `checkpoint_interval` steps (default `1000`,
§B.6), and the paper's inference scripts (`eval_noise_schedule_benchmark.py`,
per `02_schedules_noising_sampling_pdb.md`) load a checkpoint file by path —
**use the `checkpoint-ema-*` files for sampling/eval**, not the raw
`checkpoint-*` ones, unless specifically investigating training dynamics.

**Off-by-one staleness — reproduce or explicitly deviate, your call, but
know about it (§0e)**: `ema_step` is called with `loop_state.params` (params
**before** this step's `update()` call), so the EMA lags the raw checkpoint
saved at the same `step_id` by exactly one gradient step. Torch
implementation that reproduces this exactly:
```python
# at the TOP of the training step, BEFORE optimizer.step():
if ema_params is None:
    ema_params = {k: v.detach().clone() for k, v in model.state_dict().items()}
else:
    with torch.no_grad():
        for k in ema_params:
            ema_params[k].mul_(0.999).add_(model.state_dict()[k], alpha=0.001)
# ... THEN do loss.backward(); clip; optimizer.step() ...
```
If you'd rather have the more conventional "EMA of the params exactly as
they are right after this step's update" semantics, move the EMA update to
*after* `optimizer.step()` — a **deliberate, documented deviation** from
upstream, harmless in the long run (both converge to the same steady-state
tracking behavior) but not bit-for-bit reproducible against a JAX run at any
finite step count.

**Buffers vs. parameters**: `jax.tree_util.tree_map` over Haiku `params`
touches every leaf uniformly — there is no separate "buffers" concept in
Haiku the way `torch.nn.Module` distinguishes `parameters()` from
`buffers()`/non-trainable state. If the torch model has any non-trainable
buffers (e.g. non-persistent constants), they should **not** be included in
the EMA `state_dict` copy above (mirror only `model.parameters()` by name,
or filter `state_dict()` to `requires_grad` tensors) — Haiku's `params` pytree
here corresponds 1:1 to `model.parameters()`, not `model.state_dict()`
(which would additionally sweep up any buffers if present).

## B.5 Gradient accumulation, multi-GPU, checkpointing cadence and format

**Gradient accumulation** (`accumulate > 1`, dead for `default_vp`'s
defaults but specified for completeness): `simple_loop.stepgrad_accumulate`
(`flexloop/simple_loop.py:175-194`) reshapes the flat `(accumulate*size,)`
batch into `(accumulate, size, ...)`, then `jax.lax.scan`s over the leading
axis, maintaining `carry = carry + (grad - carry) / accumulate` — a
**running mean**, mathematically equal to averaging the `accumulate`
per-microbatch gradients (not a running-average-with-momentum; the
`(grad-carry)/accumulate` update rule with a **fixed** divisor `accumulate`
across all `accumulate` scan steps gives the exact unweighted mean once all
steps have executed, since it's an online-mean formula with a constant `n`
rather than an incrementing counter — verify against the fixed-`n` online
mean identity if reimplementing, but the straightforward and equivalent
torch approach is just: accumulate `.grad` via repeated
`loss.backward()` calls scaled by `1/accumulate` each, or sum raw grads and
divide once at the end). Torch idiom:
```python
optimizer.zero_grad()
for micro_batch in accumulate_chunks:            # each its own flat draw, own local batch_index (§0f)
    loss = compute_loss(model, micro_batch) / accumulate
    loss.backward()                              # grads accumulate in .grad
clip_grad_norm_(...); optimizer.step()
```

**Multi-GPU** (`multigpu=True`, `default_vp`'s actual default):
`stepgrad_pmap` (`simple_loop.py:250-269`) `pmean`s (averages across
devices) both the loss/scalar-aux and the gradient after every step;
`pmap_jit` (`:271-284`) reshapes the flat batch into
`(NUM_DEVICES, size, ...)` before dispatch. **This is `jax.pmap`
data-parallelism — the direct torch analogue is `DistributedDataParallel`
(DDP)**, which averages gradients across ranks automatically on
`loss.backward()`. **DDP requires `find_unused_parameters=True` for this
model** — per `04_PARITY_AND_PERF.md`'s training note, **202 parameters get
`grad=None`** every forward/backward pass (Haiku allocates certain `.b` bias
tensors that a zero-init/no-bias code path never actually applies in the
forward computation graph); without `find_unused_parameters=True`, DDP's
default strict-usage check will error out (or, in older DDP versions, hang
waiting for an all-reduce contribution that never comes). This flag has a
performance cost (an extra traversal to find unused params each step) but is
mandatory here, not optional.

**Checkpointing** (`simple_loop.training`, `flexloop/simple_loop.py:75-92,
348-372`; and `load_loop_state`, `:94-101`):
- **Periodic model snapshots**: every `checkpoint_interval` steps (default
  `1000`, `training()`'s own default — `train_noise_schedule_benchmark.py`
  does not override it), write two files:
  `{path}/checkpoint-{step_id}.jax` (raw params) and
  `{path}/checkpoint-ema-{step_id}.jax` (EMA params) — both via `checkpoint()`
  (`:75-77`): `pickle.dump(cast_float(item, jnp.float32), f)`. `cast_float`
  (`flexloop/loop.py:201-209`) casts every floating-point leaf to `float32`
  (a no-op if already fp32, relevant only if training ran in mixed
  precision — `default_vp`'s training step itself already explicitly casts
  data to `jnp.float32` in `model_step`, `train_noise_schedule_benchmark.py:26`).
  These accumulate forever (no rotation/cleanup logic) — **an operational
  concern for disk usage on long runs, not a training-loop correctness
  concern**; port a retention policy if desired, it isn't upstream's
  behavior but is reasonable local practice.
- **Resumable full-state snapshot**: every `save_interval` **seconds** (not
  steps — default `600` = 10 minutes, wall-clock, checked at the top of each
  training-loop iteration against a `last_time` variable,
  `save_loop_state`, `simple_loop.py:79-92`), overwrite a single
  `{path}/save.jax` containing `{params, opt_state, aux_state (= {ema_params:
  ...}), key, step_id}`, all float-cast to fp32. This is the file
  `load_loop_state` reads to resume (`:94-101`) — **note it silently
  succeeds with `None` (fresh start) on ANY exception**, including a
  corrupted/partial file from a crash mid-write — a torch port should
  probably not swallow all exceptions this broadly, but should replicate
  "if no checkpoint exists, start fresh" as the intended behavior. Suggested
  torch format: a single `torch.save(dict(model=state_dict, optimizer=...,
  scheduler_step=..., ema=ema_state_dict, step_id=..., rng_state=...),
  "save.pt")`, written atomically (write to a temp file then rename) to
  avoid the upstream's apparent vulnerability to a crash-mid-write leaving a
  truncated `save.jax`.
- **PRNG state**: the JAX `key` is part of both the periodic full-state save
  and the per-step `State` namedtuple, split fresh each step
  (`key, subkey = jax.random.split(loop_state.key)`, `simple_loop.py:357`,
  and again inside `training_inner`, `train_noise_schedule_benchmark.py:53`
  — **two independent splits per step**, one consumed by the model's
  internal `hk.next_rng_key()` calls via `subkey`, one is the carried-forward
  `key` for the next step). A torch port should likewise persist and restore
  its RNG state (`torch.get_rng_state()`/`torch.random.set_rng_state`, plus
  Python's `random` and NumPy's global RNG states used by the `Dataset`) if
  exact resume-reproducibility is wanted; this is a "nice to have," not
  required for training to function correctly after a resume.

## B.6 What is logged

`log()` (`flexloop/simple_loop.py:323-346`) is a generic TensorBoard
dispatcher: scalars (`float`/`int`/0-d or 1-d array, 1-d gets `.mean()`ed
first) go to `writer.add_scalar`, 2-3-d arrays to `add_image`, 4-d to
`add_images`; dict-valued entries with a `"marked"` key trigger a
kind-specific callback (unused by `default_vp` — no `**loggers` passed to
`log()` in the training script, so any dict-shaped loggable would silently
no-op — moot since none are produced here). What actually gets logged every
step, from `training_inner`'s `loggables` (= `aux` returned by `update()`,
`simple_loop.py:107-137`) merged with `model_step`'s `res_dict`:
- **`total`**: the full weighted loss sum (`aux["total"] = loss.mean()`,
  `simple_loop.py:127`) — this is the scalar `total` computed by `loss()` as
  specified in `03_training_losses.md` §6.
- **`{name}_loss` for every key in `out["losses"]`**
  (`train_noise_schedule_benchmark.py:31-34`): `aa_loss`, `dssp_loss`,
  `pos_loss`, `pos_trajectory_loss`, `x_loss`, `rotation_loss`,
  `rotation_trajectory_loss`, `fape_loss`, `fape_trajectory_loss`,
  `local_loss`, `distogram_loss`, `violation_loss` — i.e. every **unweighted**
  individual loss term from `03_training_losses.md` §7 (before its config
  weight is applied — the weights are only baked into `total`).
- No gradient norms, parameter norms, learning rate, or throughput metrics
  are logged to TensorBoard by this script (`update_step`'s `aux["gradients"]
  = loggable(...)`/`aux["parameters"] = loggable(...)` pattern exists in
  `flexloop/loop.py:279-280`'s **different** `update_step` — the training
  script uses `simple_loop.update_step`, which does not add these — so
  gradient/parameter histograms are **not** part of `default_vp`'s actual
  logged run). Console-only (`print`, not TensorBoard): per-step load time /
  step time / total time (`train_noise_schedule_benchmark.py:64`) and a
  full-iteration wall-clock line from `simple_loop.training`'s own loop
  (`:371`). These are throughput diagnostics — reproduce with e.g. Python
  `logging`/`print`, not required to feed TensorBoard.
- **Validation**: `simple_loop.training` accepts an optional `valid_inner`
  callback, `default_vp`'s call passes `valid_inner=None` — **there is no
  validation loop in this training script**. Any eval happens out-of-band
  via the separate `eval_noise_schedule_benchmark.py` script against saved
  checkpoints (covered in `02_schedules_noising_sampling_pdb.md`).

A reasonable torch equivalent: a `torch.utils.tensorboard.SummaryWriter`
(same library upstream uses — `train_noise_schedule_benchmark.py:10`),
`writer.add_scalar(f"{name}_loss", value, step)` for each loss-dict entry
plus `writer.add_scalar("total", total_loss, step)`, once per step; nothing
else is required for parity with what `default_vp` actually logged.

## B.7 Summary: what a torch training loop genuinely needs vs. what's incidental

**Genuinely needed to train `default_vp` at all:**
- The `BatchedProteinPDBStream`-equivalent `Dataset` (§A.3-A.4), reading
  existing `.npz` caches (§A.1) and its exact bin-packing/filtering policy.
- `assign_dssp`'s `dssp` output (§A.6.2) — required by the loss and by
  sequence/SS conditioning.
- The loss function itself (`03_training_losses.md`, separate spec).
- Plain `Adam(betas=(0.9,0.99), eps=1e-9)` + global-norm clip `0.1` +
  warmup(1000)-then-cosine-decay(500000, 1e-3→1e-5) LR schedule (§B.2-B.3).
- EMA of the parameters, decay `0.999` (§B.4), for producing usable
  checkpoints — the raw (non-EMA) weights are not what published results use.
- Periodic checkpointing so a multi-day run survives preemption (§B.5).

**Incidental to the author's specific cluster setup — implement a simpler
single-GPU equivalent, don't chase JAX-specific mechanics:**
- `hk.vmap`/`rebatch`, `jax.lax.scan`/`accumulate`-as-a-scan,
  `pmap`/`multigpu` (§0f, §B.5) — all default to `1`/off; a straightforward
  single-process, optionally-DDP torch loop covers the same ground with far
  less machinery. `find_unused_parameters=True` is the one DDP-specific
  requirement that is **not** optional (§B.5).
- Haiku's lazy parameter init / `tabulate` introspection (§B.1 step 4-5).
- `nanhunt` eager-debug mode, the `.jax` pickle checkpoint format
  specifically (any serialization format is fine as long as it round-trips
  params+optimizer+EMA+step+RNG), the exact TensorBoard log directory
  naming scheme (`{path}/salad/nsb-{config}-{num_aa}-{suffix}`).
- `flexloop`'s generic multi-step/multi-optimizer `TrainingLoop` class in
  `loop.py` — **not used by `train_noise_schedule_benchmark.py` at all**
  (it imports from `simple_loop.py` only); specified in this repo's
  `flexloop/loop.py` for a different, more general training script, and
  included in the task's file list, but confirm before assuming any of its
  mechanics (e.g. `loggable()`/`Log.loggables()`'s dict-recursion logging,
  `staged_update_step`, `alpa` multi-device support) apply to `default_vp` —
  they don't; `default_vp` uses only `simple_loop.py: training,
  update_step, rebatch_call, State` and `loop.py: cast_float`, exactly as
  the task brief anticipated.
