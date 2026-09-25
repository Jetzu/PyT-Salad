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
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from salad_torch.config import get_config  # noqa: E402
from salad_torch.convert.haiku_to_torch import load_model, load_param_npz  # noqa: E402
from salad_torch.diffusion.sampling import SamplerConfig, make_initial_data, sample  # noqa: E402
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
    ap.add_argument("--num_steps", type=int, default=500)
    ap.add_argument("--out_steps", default="400")
    ap.add_argument("--prev_threshold", type=float, default=0.8)
    ap.add_argument("--timescale_pos", default="cosine(t)")
    ap.add_argument("--cloud_std", default="")
    ap.add_argument("--out_path", default="outputs/designs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compile", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jax_seed", type=int, default=None,
                    help="Alias for --seed, for command lines copied from the "
                         "JAX CLI. No seed value gives bit-identical output "
                         "between the JAX and PyTorch backends - the two "
                         "frameworks' PRNGs are unrelated, so this only gives "
                         "reproducibility within one backend.")
    ap.add_argument("--timing_json", default="")

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
def run_renoise(step_fn, snapshot: dict, base_data: dict, cfg: SamplerConfig,
                init_prev: dict, generator) -> list[dict]:
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
            update, _ = step_fn(d, init_prev, generator)
            d["pos"] = update["pos"]
            d["seq"] = update["aa"].argmax(dim=-1)
            d["atom_pos"] = update["atom_pos"]
            d["aatype"] = update["aatype"]
            d["aa_logits"] = update["aa"]
        passes.append(d)
    return passes


# ---------------------------------------------------------------------------
# §0: the precedence chain (num_aa / dssp / template / order_agnostic)
# ---------------------------------------------------------------------------
def apply_precedence_chain(args):
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


def write_result_pdb(path: str, snapshot: dict, *, chain_index, order_agnostic: bool) -> None:
    """`snapshot_to_protein` + the §4.2 output-side reorder, then write."""
    prot = snapshot_to_protein(snapshot, chain_index=chain_index)
    if order_agnostic:
        prot = reorder_amino_acids(prot)
    with open(path, "wt") as fh:
        fh.write(to_pdb(prot))


def main():
    args = build_arg_parser().parse_args()
    if args.jax_seed is not None:
        args.seed = args.jax_seed

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

    cfg = SamplerConfig(num_steps=args.num_steps,
                        out_steps=tuple(int(x) for x in args.out_steps.split(",")),
                        prev_threshold=args.prev_threshold,
                        timescale_pos=args.timescale_pos)
    os.makedirs(args.out_path, exist_ok=True)
    if device.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"config {args.config}  L={num_aa}  steps={args.num_steps} "
          f"out_steps={cfg.out_steps}"
          + (f"  sym={sym.group}(count={sym.replicate_count})" if sym.active else ""))

    gen = torch.Generator(device=device).manual_seed(args.seed)
    stats, times = [], []
    seq_path = os.path.join(args.out_path, "sequences.fa")

    for design in range(args.num_designs):
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

        # -- Hook A: init_pos symmetrisation (§1.4), once ------------------
        if sym.active:
            data["pos"] = screw_replicate(
                data["pos"], count=sym.replicate_count, angle=sym.angle_rad,
                radius=sym.radius, translation=sym.translation,
                chain_center=False, mean=sym.mean)

        def step_fn(d, p, generator=None, _sym=sym):
            with torch.no_grad():
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
            def final_step_fn(d, p, generator=None):
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
