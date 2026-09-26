"""De-novo protein generation - the real entry point.

    python scripts/generate.py --num_aa 100 --num_designs 10 --out_path out/

Mirrors the CLI surface of upstream's
`salad/training/eval_noise_schedule_benchmark.py`, both the plain de-novo case
and the generation-loop flags specified in
`notes/specs/06_generation_flags.md` (symmetry, template conditioning,
renoise, order-agnostic decoding, dssp/dssp_mean, and the smaller
merge_chains/jax_seed/sequence_only/sample_final/depth_adjust flags).

With none of the new flags passed, behaviour is unchanged from before this
spec: one `design_{n}.pdb` per design, written from the last out-step.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch
from torch import Tensor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from salad_torch import opt as salad_opt  # noqa: E402
from salad_torch.config import get_config  # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.diffusion.sampling import (  # noqa: E402
    SamplerConfig, StepFn, make_initial_data, sample, sample_group)
from salad_torch.diffusion.schedules import parse_num_aa, parse_timescale  # noqa: E402
from salad_torch.inference.options import decode_sequence, parse_dssp, parse_dssp_mean  # noqa: E402
from salad_torch.inference.postprocess import reorder_amino_acids  # noqa: E402
from salad_torch.inference.symmetry import parse_sym, screw_replicate  # noqa: E402
from salad_torch.inference.template import parse_template  # noqa: E402
from salad_torch.io.pdb import snapshot_to_protein, to_pdb  # noqa: E402
from salad_torch.modules.model import StructureDiffusionInference, apply_diffusion  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default_vp")
    ap.add_argument("--params", default="")
    ap.add_argument("--num_aa", default="100")
    ap.add_argument("--num_designs", type=int, default=1)
    # None means "not given": a preset may then supply its own schedule
    # (see PRESETS[...]["steps_scale"]). An explicit value always wins.
    ap.add_argument("--num_steps", type=int, default=None)
    ap.add_argument("--out_steps", default=None)
    ap.add_argument("--prev_threshold", type=float, default=0.8)
    ap.add_argument("--timescale_pos", default="cosine(t)")
    ap.add_argument("--cloud_std", default="")
    ap.add_argument("--out_path", default="outputs/designs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compile", default="")
    ap.add_argument("--fast", action="store_true",
                    help="use the compiled + CUDA-graph inference path "
                         "(~15x faster; GPU only, fixed length, no --sym)")
    ap.add_argument("--preset", default="exact",
                    help="which --fast inference preset to run. 'exact' (the "
                         "default) holds the port's measured parity floor "
                         "against JAX salad; any other preset is allowed to "
                         "trade numerics for speed. See "
                         "salad_torch.modules.fast.PRESETS.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jax_seed", type=int, default=None,
                    help="Alias for --seed, for command lines copied from the "
                         "JAX CLI. No seed value gives bit-identical output "
                         "between the JAX and PyTorch backends - the two "
                         "frameworks' PRNGs are unrelated, so this only gives "
                         "reproducibility within one backend.")
    ap.add_argument("--timing_json", default="")
    ap.add_argument("--group_size", default="",
                    help="N019. Advance N INDEPENDENT designs through ONE "
                         "CUDA graph instead of one at a time. Needs --fast. "
                         "Not a batch axis and not a shared trajectory: each "
                         "design keeps its own buffers, its own noise and its "
                         "own seed-derived RNG stream, and computes exactly "
                         "what it would compute alone - the only thing that "
                         "changes is that the GPU stops idling between one "
                         "design's serialised kernel launches. Unset (the "
                         "default) keeps the existing one-design-at-a-time "
                         "loop byte-for-byte. 'auto' picks N from THIS "
                         "device's SM count and free VRAM "
                         "by MEASURING it on this device - it captures the "
                         "graph at 1, 2, 3, 4, ... and stops when a rung "
                         "stops paying, which costs seconds against the "
                         "one-time compile; an "
                         "integer pins it; 1 is the grouped machinery with a "
                         "group of one, which is the reference a grouped run "
                         "must reproduce design-for-design. The win comes "
                         "from idle SMs, so it is largest on a big card with "
                         "a short protein and shrinks towards 1x on a small "
                         "card or a long one.")

    # -- §1: symmetry -------------------------------------------------
    sym = ap.add_argument_group("symmetry (notes/specs/06_generation_flags.md §1)")
    sym.add_argument("--sym", default="",
                     help="Screw/cyclic symmetry group. '' disables it. "
                          "'cN' (e.g. 'c4') is a port-side convenience that "
                          "sets --replicate_count=N, --screw_angle=360/N and "
                          "--screw_translation=0 as defaults (overridable). "
                          "Requires --num_aa given as N colon-separated, "
                          "ideally equal-length, chains.")
    sym.add_argument("--sym_mean", default="False", choices=["True", "False"],
                     help="'True'/'False' as a literal string, matching "
                          "upstream's `mean=(sym_mean == 'True')`. Controls "
                          "the init_pos and per-step post-hoc symmetrisers; "
                          "the noised-input symmetriser always uses False "
                          "regardless (an upstream inconsistency, reproduced).")
    sym.add_argument("--sym_threshold", type=float, default=0.0,
                     help="Post-hoc per-step symmetrisation only runs while "
                          "raw_t > this. 0.0 (default) symmetrises after "
                          "every step; raising it (e.g. 0.5) relaxes the "
                          "constraint for late, low-noise refinement.")
    sym.add_argument("--replicate_count", type=int, default=None,
                     help="Number of subunits ('count'). Defaults from 'cN' "
                          "in --sym if not given.")
    sym.add_argument("--screw_angle", type=float, default=None,
                     help="Degrees, about the Y axis. Defaults from 'cN' in "
                          "--sym (360/N) if not given.")
    sym.add_argument("--screw_radius", type=float, default=0.0,
                     help="Offset of the representative subunit from the "
                          "screw/rotation axis before replication.")
    sym.add_argument("--screw_translation", type=float, default=0.0,
                     help="Rise per subunit along Y. 0 = plain cyclic (Cn) "
                          "point-group symmetry; nonzero = true screw "
                          "(helical) symmetry.")

    # -- §5.4 / §0: dssp, merge_chains ---------------------------------
    ap.add_argument("--dssp", default="none",
                    help="'none', 'random', or a literal 'LHE_' string. A "
                         "literal string's length OVERRIDES --num_aa and "
                         "discards --merge_chains/cyclic num_aa specs (§0).")
    ap.add_argument("--dssp_mean", default="none",
                    help="'none', 'random', or 'loop,helix,sheet' (not "
                         "renormalised - must already sum to 1).")
    ap.add_argument("--merge_chains", action="store_true",
                    help="Zero out chain_index before the model sees it (a "
                         "no-op if --dssp <string> or --template is also "
                         "given, both of which reassign chain afterwards).")

    # -- §5.2: sequence_only --------------------------------------------
    ap.add_argument("--sequence_only", action="store_true",
                    help="Write FASTA (appended to {out_path}/sequences.fa) "
                         "instead of PDB files.")

    # -- §2: template -----------------------------------------------------
    ap.add_argument("--template", default="",
                    help="'<pdb_path>,<constring>'. constring is one char "
                         "per residue plus ':' chain separators; 'F' = fixed/"
                         "conditioned on the template, anything else = free. "
                         "Overrides --num_aa/--dssp entirely and disables "
                         "cyclic conditioning (§0, §2).")
    ap.add_argument("--template_aa", action="store_true",
                    help="Also condition on the template's sequence identity "
                         "(aa_condition), not just its geometry (dmap/omap).")

    # -- §4 / §3 / §5.3 / §5.5 --------------------------------------------
    ap.add_argument("--order_agnostic", action="store_true",
                    help="Network never sees residue order (resi=-1, "
                         "chain=0); output PDB is re-threaded by following "
                         "C-N backbone contacts instead of network indexing.")
    ap.add_argument("--renoise", action="store_true",
                    help="At every out-step, additionally run 9 independent "
                         "partial re-denoise passes (self-conditioning "
                         "forced off) re-entering at raw_t=0.4. EXPENSIVE: "
                         "with the CLI defaults this is roughly a 3x "
                         "wall-clock multiplier per design (§3.3). Switches "
                         "PDB naming to result_{design}_{idr}_{step}.pdb.")
    ap.add_argument("--sample_final", action="store_true",
                    help="On out-steps only, replace the ADM single-shot "
                         "decode with an L-step categorical random-order "
                         "autoregressive re-decode of the sequence "
                         "(AADecoder.sample), then rebuild sidechains from "
                         "it. Costs L extra decoder-only forward passes per "
                         "out-step.")
    ap.add_argument("--sample_final_temperature", type=float, default=0.1,
                    help="Softmax temperature for --sample_final. Upstream "
                         "hardcodes 0.1 with no CLI flag; this default "
                         "matches it exactly, the flag is a port-side "
                         "addition.")
    ap.add_argument("--depth_adjust", default="none",
                    help="'none' or an int N <= the checkpoint's native "
                         "depth (6 for default_vp): early-exit ablation, "
                         "only the first N trunk blocks are built and "
                         "loaded. N > native depth is not supported (no "
                         "extra trained weights to load).")
    return ap


# ---------------------------------------------------------------------------
# §3: --renoise
# ---------------------------------------------------------------------------
def run_renoise(step_fn: StepFn, snapshot: dict, base_data: dict, cfg: SamplerConfig,
                init_prev: dict, generator: torch.Generator | None) -> list[dict]:
    """9 independent partial re-denoise passes from one out-step snapshot. §3.

    Each pass operates on its own copy of `snapshot`'s tensors - deliberately
    NOT the shared, mutated-in-place dict upstream uses (§3.2's aliasing bug:
    a `for` loop variable outliving its loop silently forks the rest of the
    main trajectory through the 9th pass's output whenever `--out_steps`
    names more than one step). That bug is not reproduced; the caller's
    `snapshot`/main trajectory are never touched by this function.

    `base_data` supplies the conditioning/setup keys that never change over
    the run (`aa_gt`, and whichever of `aa_condition` / `dssp_condition` /
    `dssp_mean(+mask)` / `dmap`+`omap`+`dmap_mask` / `cloud_std` are present) -
    `sample()`'s snapshots deliberately only carry the per-step-varying
    fields (`_SNAPSHOT_KEYS`), so `prepare_condition` would otherwise KeyError
    on `aa_gt` the moment a renoise sub-step calls the model. `sample()` never
    mutates the dict passed to it, so the caller's original `data` is exactly
    this "never changes" set and can be passed through unchanged.

    Always the base `step_fn`, never `final_step_fn` - `--sample_final` has
    no effect inside a renoise pass. `--prev_threshold` likewise has no
    effect: self-conditioning is unconditionally off (`prev = init_prev`)
    throughout every sub-step.
    """
    timescale_pos = parse_timescale(cfg.timescale_pos)
    timescale_seq = (None if cfg.timescale_seq is None
                     else parse_timescale(cfg.timescale_seq))
    denominator = cfg.raw_t_denominator or cfg.num_steps
    step_target = snapshot["step"]
    start = int(cfg.num_steps * 0.6)
    skip_keys = ("step", "raw_t", "result")

    passes = []
    for _ in range(9):
        d = dict(base_data)
        d.update({k: v.clone() for k, v in snapshot.items()
                 if k not in skip_keys and torch.is_tensor(v)})
        # if step_target <= start, this range is empty: the pass is an
        # identical copy of the un-renoised snapshot, not an error (§3.1)
        for s in range(start, step_target):
            raw_t = 1 - s / denominator
            scaled_t_pos = timescale_pos(raw_t)
            scaled_t_seq = 1.0 if timescale_seq is None else timescale_seq(raw_t)
            d["t_pos"] = scaled_t_pos * torch.ones_like(d["t_pos"])
            d["t_seq"] = scaled_t_seq * torch.ones_like(d["t_seq"])
            # P2: every sub-step but the last overwrites keys nothing reads
            need_aa = (not cfg.lean_steps) or s == step_target - 1
            update, _ = step_fn(d, init_prev, generator, need_aa=need_aa)
            d["pos"] = update["pos"]
            if "aa" in update:
                d["seq"] = update["aa"].argmax(dim=-1)
                d["aa_logits"] = update["aa"]
            if "atom_pos" in update:
                d["atom_pos"] = update["atom_pos"]
            if "aatype" in update:
                d["aatype"] = update["aatype"]
        passes.append(d)
    return passes


# ---------------------------------------------------------------------------
# §0: the precedence chain (num_aa / dssp / template / order_agnostic)
# ---------------------------------------------------------------------------
def apply_precedence_chain(args: argparse.Namespace) -> dict:
    """Reproduce §0's fixed-order overwrites. Returns a dict of everything the
    per-design loop needs: num_aa, resi, chain, base_chain, is_cyclic,
    cyclic_mask, plus optional dssp/dssp_mean/template payloads.
    """
    num_aa, resi, chain, is_cyclic, cyclic_mask = parse_num_aa(args.num_aa)
    base_chain = chain.clone()

    if args.merge_chains:
        chain = torch.zeros_like(chain)

    dssp_fixed, dssp_fn = None, None
    dssp_text = args.dssp.strip()
    if dssp_text not in ("none",):
        parsed = parse_dssp(dssp_text)
        if callable(parsed):
            dssp_fn = parsed
        elif parsed is not None:
            # a literal string: silently discards --num_aa/--merge_chains/cyclic
            dssp_fixed = parsed
            num_aa = dssp_fixed.shape[0]
            resi = torch.arange(num_aa, dtype=torch.long)
            base_chain = chain = torch.zeros(num_aa, dtype=torch.long)
            is_cyclic = False
            cyclic_mask = torch.zeros(num_aa, dtype=torch.bool)

    dssp_mean_fixed, dssp_mean_fn = None, None
    dssp_mean_text = args.dssp_mean.strip()
    if dssp_mean_text not in ("none",):
        parsed_mean = parse_dssp_mean(dssp_mean_text)
        if callable(parsed_mean):
            dssp_mean_fn = parsed_mean
        elif parsed_mean is not None:
            dssp_mean_fixed = parsed_mean

    template = None
    if args.template:
        aa_condition, dmap, omap, dmap_mask, t_chain = parse_template(args.template)
        num_aa = dmap.shape[0]
        resi = torch.arange(num_aa, dtype=torch.long)
        base_chain = chain = t_chain
        is_cyclic = False
        cyclic_mask = torch.zeros(num_aa, dtype=torch.bool)
        template = dict(aa_condition=aa_condition, dmap=dmap, omap=omap,
                        dmap_mask=dmap_mask)

    if args.order_agnostic:
        resi = -torch.ones(num_aa, dtype=torch.long)
        chain = torch.zeros(num_aa, dtype=torch.long)
        # base_chain is NOT touched here - see §4.1

    return dict(num_aa=num_aa, resi=resi, chain=chain, base_chain=base_chain,
               is_cyclic=is_cyclic, cyclic_mask=cyclic_mask,
               dssp_fixed=dssp_fixed, dssp_fn=dssp_fn,
               dssp_mean_fixed=dssp_mean_fixed, dssp_mean_fn=dssp_mean_fn,
               template=template)


def make_design_data(args: argparse.Namespace, pc: dict, num_aa: int,
                     resi: Tensor, chain: Tensor, is_cyclic: bool,
                     cyclic_mask: Tensor, device: torch.device
                     ) -> tuple[dict, dict]:
    """One design's ``(data, init_prev)``: the per-design body of the loop.

    Lifted out verbatim so the grouped path (--group_size) builds its N
    designs through exactly the code the ordinary loop builds its one with.
    Order matters and is preserved: `--dssp random` / `--dssp_mean random`
    draw from the stdlib RNG here, so calling this once per design in design
    order reproduces the sequence of draws the ordinary loop makes.
    """
    data, prev = make_initial_data(
        num_aa, resi, chain, cyclic_mask if is_cyclic else None, device=device)
    if args.cloud_std:
        data["cloud_std"] = torch.full((num_aa,), float(args.cloud_std), device=device)

    if pc["dssp_fixed"] is not None:
        data["dssp_condition"] = pc["dssp_fixed"].to(device)
    elif pc["dssp_fn"] is not None:
        data["dssp_condition"] = pc["dssp_fn"](num_aa).to(device)
    if pc["dssp_mean_fixed"] is not None:
        data["dssp_mean"] = pc["dssp_mean_fixed"].to(device)
    elif pc["dssp_mean_fn"] is not None:
        data["dssp_mean"] = pc["dssp_mean_fn"]().to(device)
    if pc["template"] is not None:
        data["dmap"] = pc["template"]["dmap"].to(device)
        data["omap"] = pc["template"]["omap"].to(device)
        data["dmap_mask"] = pc["template"]["dmap_mask"].to(device)
        if args.template_aa:
            data["aa_condition"] = pc["template"]["aa_condition"].to(device)
    return data, prev


def design_generator(device: torch.device, seed: int, design: int
                     ) -> torch.Generator:
    """The RNG stream design `design` gets in a grouped run.

    The ordinary loop runs every design off ONE generator in sequence, so
    design 1's noise depends on how many draws design 0 made. That ordering
    cannot be preserved when designs advance in lockstep, so the grouped path
    gives each design an independent, seed-derived stream instead. The
    consequence is the property the whole feature is gated on: design `i` of a
    ``--group_size 8`` run and design `i` of a ``--group_size 1`` run are the
    same design, because they are fed the same stream.

    The multiplier is an odd constant purely so consecutive seeds do not hand
    consecutive designs overlapping streams; there is nothing tuned about it.
    """
    g = torch.Generator(device=device)
    g.manual_seed((int(seed) * 1_000_003 + int(design)) & 0x7FFF_FFFF_FFFF_FFFF)
    return g


def write_result_pdb(path: str, snapshot: dict, *, chain_index: Tensor,
                     order_agnostic: bool) -> None:
    """`snapshot_to_protein` + the §4.2 output-side reorder, then write.

    `--order_agnostic` gives the network zero sequence-order information
    (`resi=-1` for every residue, §4.1), and measured verification runs of
    this port show that at least with the `default_vp-200k` checkpoint this
    quite often produces a backbone with a handful of residues that never
    form a clean peptide-bond-like (< 2.5 A C-N) link to the rest of the
    chain (2-11 of 32 residues, unrelated to how many diffusion steps are
    run: 30 through 300 all reproduce it). `reorder_amino_acids` is correctly
    unable to thread such residues into a walk and raises `RuntimeError`
    (its documented, deliberate replacement for upstream's unbounded hang -
    see its docstring) rather than fabricate a connection that isn't there.
    That is a real property of order-agnostic generation with this
    checkpoint, not a bug in the reordering code - but a single malformed
    design should not abort a whole `--num_designs` run. Fall back to
    writing the network's own (unordered, un-re-threaded) residue sequence
    with a warning, rather than losing the design entirely.

    The fallback also has to repair `residue_index`: `--order_agnostic` feeds
    the network `resi = -1` for *every* residue (§4.1, so the network gets no
    sequence-order signal at all), and that same all-``-1`` tensor is what
    `make_initial_data` stores as `data["residue_index"]` - the field this
    function's normal path relies on `reorder_amino_acids` to overwrite with
    real, unique numbers (`new_residue_index`, from the walk order). Skip that
    overwrite and every residue is written with the *same* PDB residue number
    -1 in one chain; `to_pdb` doesn't reject that, but a standard reader
    (including this port's own `read_pdb`) silently merges same-numbered
    residues in the same chain on read-back, silently losing residues. A
    plain `0..L-1` renumbering is not the geometric thread reorder_amino_acids
    would have produced, but it is a well-formed, round-trippable PDB.
    """
    prot = snapshot_to_protein(snapshot, chain_index=chain_index)
    if order_agnostic:
        try:
            prot = reorder_amino_acids(prot)
        except RuntimeError as exc:
            print(f"  warning: --order_agnostic reorder failed for {path} "
                  f"({exc}); writing the network's own residue order instead.")
            prot = dataclasses.replace(
                prot, residue_index=np.arange(prot.residue_index.shape[0],
                                              dtype=prot.residue_index.dtype))
    with open(path, "wt") as fh:
        fh.write(to_pdb(prot))


def run_grouped(args, config, model, cfg: SamplerConfig, pc: dict, num_aa: int,
                resi: Tensor, chain: Tensor, base_chain: Tensor,
                is_cyclic: bool, cyclic_mask: Tensor,
                device: torch.device) -> None:
    """`--group_size`: N designs per captured graph (N019).

    The design-facing behaviour is the ordinary one - ``design_{n}.pdb``, the
    same stats line, the same `--timing_json` - and the only thing that
    differs is that `--num_designs` is consumed in groups instead of one at a
    time. A trailing partial group is run as its own, smaller group rather
    than padded with throwaway designs, so `--num_designs 10 --group_size 4`
    costs 4+4+2 and writes 10 designs, not 12.
    """
    from salad_torch.modules.fast import GroupFastDenoiser

    if not args.fast:
        raise SystemExit("--group_size needs --fast (it groups the captured "
                         "graph; there is nothing to group on the eager path)")
    if device.type != "cuda":
        raise SystemExit("--group_size needs a CUDA device")
    for flag in ("sym", "renoise", "sequence_only", "sample_final"):
        if getattr(args, flag) not in ("", False, None):
            raise SystemExit(
                f"--group_size is incompatible with --{flag}: it steps every "
                f"design of the group through one captured replay, and "
                f"--{flag} inserts per-design host work between the noising "
                f"and the denoising that the capture cannot see. Drop "
                f"--group_size to get the ordinary path back.")

    want = args.group_size.strip().lower()
    if want != "auto":
        try:
            want = int(want)
        except ValueError:
            raise SystemExit(f"--group_size must be 'auto' or an integer, "
                             f"got {args.group_size!r}")
        if want < 1:
            raise SystemExit("--group_size must be >= 1")

    t_warm = time.perf_counter()
    pc_hint = bool(config.multi_motif or pc["template"] is not None
                   or int(chain.max()) > 0)
    group = GroupFastDenoiser(model.predict, num_aa, group_size=want,
                              device=device, preset=args.preset).warmup(
                                  pair_cond=pc_hint)
    n = group.group_size
    peak_gb = (group.peak_bytes or 0) / 2 ** 30
    per_gb = (group.per_member_bytes or 0) / 2 ** 30
    if want == "auto":
        props = torch.cuda.get_device_properties(device)
        print(f"group_size auto -> {n} on {props.name} "
              f"({props.multi_processor_count} SMs, "
              f"{props.total_memory / 2**30:.1f} GiB)")
    print(f"grouped fast path ({args.preset}, group_size={n}) compiled + "
          f"captured in {time.perf_counter() - t_warm:.0f}s (one-time); "
          f"peak reserved {peak_gb:.2f} GiB, ~{per_gb:.3f} GiB/design")
    if n == 1:
        print("  group_size 1: this is the ordinary single-design "
              "computation, run through the grouped machinery - the "
              "reference a larger group must reproduce.")

    os.makedirs(args.out_path, exist_ok=True)
    stats, times, group_wall = [], [], []
    written = 0
    for start in range(0, args.num_designs, n):
        members = list(range(start, min(start + n, args.num_designs)))
        if len(members) != n:
            # trailing partial group: recapture at the smaller size rather
            # than run throwaway designs. One extra capture, not one extra
            # compile - the compiled forward is shared.
            group.resize(len(members), pair_cond=pc_hint)
        group.invalidate_condition_cache()
        datas, prevs, gens = [], [], []
        for d in members:
            data, prev = make_design_data(args, pc, num_aa, resi, chain,
                                          is_cyclic, cyclic_mask, device)
            datas.append(data)
            prevs.append(prev)
            gens.append(design_generator(device, args.seed, d))
        gate = group.pin_conditions(datas)

        def step_fn_group(ds, ps, gs, *, need_aa=True, _gate=gate):
            with torch.no_grad():
                noised = [apply_diffusion(config, d, generator=g)
                          for d, g in zip(ds, gs)]
                fed = [dict(d, **nz) for d, nz in zip(ds, noised)]
                return group(fed, ps, need_aa=need_aa, pair_cond=_gate,
                             generators=gs)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        snaps_per_design = sample_group(step_fn_group, datas, prevs, cfg,
                                        generators=gens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        # one entry PER DESIGN, not per group - the summary line below sums
        # this to get the run's total GPU time, and a per-group entry made a
        # group of 8 report a total of one design's worth.
        times.extend([elapsed / len(members)] * len(members))
        group_wall.append(elapsed)

        for d, snaps in zip(members, snaps_per_design):
            snap = {k: (v.cpu() if torch.is_tensor(v) else v)
                    for k, v in snaps[-1].items()}
            path = os.path.join(args.out_path, f"design_{d}.pdb")
            write_result_pdb(path, snap, chain_index=base_chain,
                             order_agnostic=args.order_agnostic)
            ca = snap["pos"][:, 1].numpy()
            d_ca = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
            rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean()))
            stats.append(dict(ca_ca_mean=float(d_ca.mean()),
                              ca_ca_min=float(d_ca.min()),
                              ca_ca_max=float(d_ca.max()), rg=rg,
                              seconds=elapsed / len(members)))
            written += 1
            print(f"  design {d}: {elapsed / len(members):6.1f}s/design  "
                  f"CA-CA {d_ca.mean():.3f} A (min {d_ca.min():.3f} "
                  f"max {d_ca.max():.3f})  Rg {rg:.1f} A  -> {path}")
        print(f"  group of {len(members)}: {elapsed:.1f}s wall "
              f"({elapsed / len(members):.3f}s/design)")

    t = np.array(times)
    print(f"\n{written} designs of {num_aa} aa in groups of up to {n}: "
          f"median {np.median(t):.3f}s/design, "
          f"{sum(group_wall):.1f}s of GPU time for all {written} "
          f"(compile/capture excluded)")
    if args.timing_json:
        with open(args.timing_json, "w") as fh:
            json.dump(dict(backend="torch", config=args.config, num_aa=num_aa,
                           num_steps=args.num_steps, device=str(device),
                           compile=args.compile, group_size=n,
                           peak_reserved_bytes=group.peak_bytes,
                           per_member_bytes=group.per_member_bytes,
                           stats=stats), fh, indent=2)


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.jax_seed is not None:
        args.seed = args.jax_seed

    # seed EVERY source, not just torch: inference/options.py uses the stdlib
    # `random` for --dssp/--dssp_mean "random", so --seed alone left those
    # unreproducible (measured 15-25 A between reruns).
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    config = get_config(args.config)
    if args.depth_adjust != "none":
        # must happen before the model is built AND before params are loaded
        # (§5.5): DiffusionStack's length and the checkpoint layer-slicing
        # both read config.diffusion_depth at that point.
        config.diffusion_depth = int(args.depth_adjust)
    params = args.params or f"/hnvme/workspace/v181bb10-salad/{args.config}-200k.npz"

    model = StructureDiffusionInference(config).eval().to(device)
    load_model(model.predict, load_param_npz(params), config, strict=True)
    if args.compile:
        blocks = model.predict.diffusion.diffusion_stack.blocks
        for i in range(len(blocks)):
            blocks[i] = torch.compile(blocks[i], mode=args.compile, dynamic=False)

    pc = apply_precedence_chain(args)
    num_aa = pc["num_aa"]
    resi, chain, base_chain = pc["resi"], pc["chain"], pc["base_chain"]
    is_cyclic, cyclic_mask = pc["is_cyclic"], pc["cyclic_mask"]

    sym = parse_sym(args.sym, replicate_count=args.replicate_count,
                    screw_angle=args.screw_angle, screw_radius=args.screw_radius,
                    screw_translation=args.screw_translation,
                    sym_mean=args.sym_mean, sym_threshold=args.sym_threshold)

    # A preset may shorten the trajectory (N011's "fast" halves it). Scaling
    # num_steps and out_steps TOGETHER is what makes that a coarsening rather
    # than a different run: the schedule is `raw_t = 1 - step/num_steps`, so
    # both endpoints (raw_t 1.0 at the start, 0.2 at the reported step) are
    # preserved and only the spacing changes. Skipped entirely the moment the
    # command line names either value.
    num_steps, out_steps_s = args.num_steps, args.out_steps
    steps_scale = 1.0
    if args.fast:
        from salad_torch.modules.fast import PRESETS as _PRESETS
        steps_scale = float(_PRESETS.get(args.preset, {}).get("steps_scale", 1.0))
    if steps_scale != 1.0 and num_steps is None and out_steps_s is None:
        num_steps = max(1, int(round(500 * steps_scale)))
        out_steps_s = ",".join(str(max(0, int(round(int(x) * steps_scale))))
                               for x in "400".split(","))
        print(f"preset {args.preset!r}: schedule scaled by {steps_scale} -> "
              f"num_steps={num_steps} out_steps={out_steps_s}")
    if num_steps is None:
        num_steps = 500
    if out_steps_s is None:
        out_steps_s = "400"
    args.num_steps, args.out_steps = num_steps, out_steps_s
    cfg = SamplerConfig(num_steps=args.num_steps,
                        out_steps=tuple(int(x) for x in args.out_steps.split(",")),
                        prev_threshold=args.prev_threshold,
                        timescale_pos=args.timescale_pos,
                        # P2: ask for the sequence heads only on reporting
                        # steps. Gated on `diffuse_sequence` because that is
                        # the one config key that would let `data["seq"]`
                        # re-enter the denoiser (no released checkpoint sets
                        # it). See SamplerConfig.lean_steps.
                        lean_steps=bool(args.fast and not config.diffuse_sequence
                                        and salad_opt.LEAN_STEPS))
    os.makedirs(args.out_path, exist_ok=True)
    if device.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"config {args.config}  L={num_aa}  steps={args.num_steps} "
          f"out_steps={cfg.out_steps}"
          + (f"  sym={sym.group}(count={sym.replicate_count})" if sym.active else ""))

    if args.group_size:
        run_grouped(args, config, model, cfg, pc, num_aa, resi, chain,
                    base_chain, is_cyclic, cyclic_mask, device)
        return

    fast_denoiser = None
    if args.fast:
        if device.type != "cuda":
            raise SystemExit("--fast needs a CUDA device")
        if sym.active:
            # the symmetriser mutates pos_noised between noising and denoising,
            # which the captured graph cannot see
            raise SystemExit("--fast is incompatible with --sym")
        from salad_torch.modules.fast import FastDenoiser
        t_warm = time.perf_counter()
        # HINT only (which graphs to pre-build): FastDenoiser re-derives the
        # real gate from the conditioning tensors on the first step.
        pc_hint = bool(config.multi_motif or pc["template"] is not None
                       or int(chain.max()) > 0)
        fast_denoiser = FastDenoiser(model.predict, num_aa, device=device,
                                     preset=args.preset).warmup(pair_cond=pc_hint)
        print(f"fast path ({args.preset}) compiled + captured in "
              f"{time.perf_counter() - t_warm:.0f}s (one-time)")

    gen = torch.Generator(device=device).manual_seed(args.seed)
    stats, times = [], []
    seq_path = os.path.join(args.out_path, "sequences.fa")

    for design in range(args.num_designs):
        # the conditioning is redrawn per design (--dssp random, --dssp_mean
        # random, a per-design template), so the cached `prepare_condition`
        # output from the previous design must go. Unconditional: the "exact"
        # preset re-derives its key anyway and this only saves it the work,
        # while the "fast" preset TRUSTS this call.
        if fast_denoiser is not None:
            fast_denoiser.invalidate_condition_cache()
        data, prev = make_design_data(args, pc, num_aa, resi, chain,
                                      is_cyclic, cyclic_mask, device)

        # -- Hook A: init_pos symmetrisation (§1.4), once ------------------
        if sym.active:
            data["pos"] = screw_replicate(
                data["pos"], count=sym.replicate_count, angle=sym.angle_rad,
                radius=sym.radius, translation=sym.translation,
                chain_center=False, mean=sym.mean)

        def step_fn(d, p, generator=None, *, need_aa=True,
                    _sym=sym, _fast=fast_denoiser):
            with torch.no_grad():
                if _fast is not None:
                    # the fast path denoises only, so noise explicitly first
                    noised = apply_diffusion(config, d, generator=generator)
                    # thread the seeded generator into the Gumbel neighbour
                    # draw too: without it that draw came off the DEFAULT CUDA
                    # RNG, so --seed did not actually pin a --fast run and two
                    # identical command lines produced different proteins.
                    return _fast(dict(d, **noised), p, need_aa=need_aa,
                                 generator=generator)
                if not _sym.active:
                    return model(d, p, generator=gen)
                # Hook B (§1.5): compose apply_diffusion + model.predict
                # directly instead of the fused model(...) call, so
                # sym_noise can be inserted on pos_noised. mean=False here
                # is hardcoded (not --sym_mean) - reproducing an upstream
                # inconsistency, see symmetry.py's module docstring.
                noised = apply_diffusion(config, d, generator=generator)
                d = dict(d, **noised)
                d["pos_noised"] = screw_replicate(
                    d["pos_noised"], count=_sym.replicate_count,
                    angle=_sym.angle_rad, radius=_sym.radius,
                    translation=_sym.translation, chain_center=False, mean=False)
                return model.predict(d, p)

        def post_step_fn(pos, raw_t, _sym=sym):
            # Hook C (§1.6): post-hoc per-step symmetrisation of the x0
            # prediction, gated by --sym_threshold. The repulsion nudge
            # (chain_center) only fires for raw_t > 0.5 and scales with raw_t.
            if not _sym.active or raw_t <= _sym.threshold:
                return pos
            return screw_replicate(
                pos, count=_sym.replicate_count, angle=_sym.angle_rad,
                radius=_sym.radius, translation=_sym.translation,
                chain_center=(raw_t > 0.5), time=raw_t, mean=_sym.mean)

        final_step_fn = None
        if args.sample_final:
            def final_step_fn(d, p, generator=None, *, need_aa=True):
                # §5.3: no separate "second model" - config.sample_aa is
                # never consumed by Diffusion.forward (it doesn't exist as a
                # branch there, and diffusion.py is off-limits for this
                # task), so instead we run the ordinary trunk+ADM pass to
                # get (local, pos), then override the sequence with the
                # genuine AADecoder.sample autoregressive decode and rebuild
                # sidechains from it. Net effect on out-steps is identical to
                # upstream's "second model with sample_aa=True".
                with torch.no_grad():
                    result, new_prev = model(d, p, generator=gen)
                    diffusion = model.predict.diffusion
                    aatype = diffusion.aa_decoder.sample(
                        d["aa_gt"], result["local"], result["pos"],
                        d["residue_index"], d["chain_index"], d["batch_index"],
                        d["mask"], temperature=args.sample_final_temperature,
                        generator=gen)
                    logits, _ = diffusion.aa_decoder(
                        aatype, result["local"], result["pos"],
                        d["residue_index"], d["chain_index"], d["batch_index"],
                        d["mask"])
                    result["aa"] = torch.log_softmax(logits, dim=-1)
                    result["aatype"] = aatype
                    _, _, angle_pos = diffusion.angle_head(
                        aatype, result["local"], result["pos"])
                    result["atom_pos"] = (result["pos"] if config.encode_atom14
                                          else angle_pos)
                    return result, new_prev

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        snaps = sample(step_fn, data, prev, cfg, generator=gen,
                       post_step_fn=post_step_fn if sym.active else None,
                       final_step_fn=final_step_fn)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        use_extended_naming = args.renoise or args.sequence_only
        if not use_extended_naming:
            snap = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in snaps[-1].items()}
            path = os.path.join(args.out_path, f"design_{design}.pdb")
            write_result_pdb(path, snap, chain_index=base_chain,
                             order_agnostic=args.order_agnostic)
            ca = snap["pos"][:, 1].numpy()
            d_ca = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
            rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean()))
            stats.append(dict(ca_ca_mean=float(d_ca.mean()), ca_ca_min=float(d_ca.min()),
                              ca_ca_max=float(d_ca.max()), rg=rg, seconds=elapsed))
            print(f"  design {design}: {elapsed:6.1f}s  CA-CA {d_ca.mean():.3f} A "
                  f"(min {d_ca.min():.3f} max {d_ca.max():.3f})  Rg {rg:.1f} A  -> {path}")
        else:
            # §3.1/§5.2 naming: result_{design}_{idr}_{step}.pdb (idr 0 is the
            # un-renoised original), or >design_{design}_{idr}_{step} FASTA
            # headers appended to one shared sequences.fa.
            for snap in snaps:
                step = snap["step"]
                results = [{k: (v.cpu() if torch.is_tensor(v) else v)
                          for k, v in snap.items()}]
                if args.renoise:
                    # `prev` here is still exactly `init_prev`: sample() never
                    # mutates the caller's `prev`/`data` variables (it copies
                    # them internally), so both remain the pre-loop state.
                    results += [{k: (v.cpu() if torch.is_tensor(v) else v)
                               for k, v in d.items()}
                              for d in run_renoise(step_fn, snap, data, cfg, prev, gen)]
                for idr, res in enumerate(results):
                    if args.sequence_only:
                        seq = decode_sequence(res["aatype"])
                        with open(seq_path, "at") as fh:
                            fh.write(f">design_{design}_{idr}_{step}\n{seq}\n")
                            fh.flush()
                    else:
                        path = os.path.join(args.out_path,
                                            f"result_{design}_{idr}_{step}.pdb")
                        write_result_pdb(path, res, chain_index=base_chain,
                                         order_agnostic=args.order_agnostic)
            print(f"  design {design}: {elapsed:6.1f}s  "
                  f"{len(snaps)} out-step(s)"
                  + (f" x 10 (1 original + 9 renoise)" if args.renoise else "")
                  + f"  -> {args.out_path}")
            stats.append(dict(seconds=elapsed))

    t = np.array(times)
    print(f"\n{args.num_designs} designs of {num_aa} aa: "
          f"median {np.median(t):.1f}s, total {t.sum():.1f}s")
    if args.timing_json:
        with open(args.timing_json, "w") as fh:
            json.dump(dict(backend="torch", config=args.config, num_aa=num_aa,
                           num_steps=args.num_steps, device=str(device),
                           compile=args.compile, stats=stats), fh, indent=2)


if __name__ == "__main__":
    main()
