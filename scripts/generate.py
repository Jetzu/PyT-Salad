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
    SamplerConfig, StepFn, make_initial_data, pack_designs, sample,
    sample_group, split_packed_snapshot)
from salad_torch.diffusion.schedules import parse_num_aa, parse_timescale  # noqa: E402
from salad_torch.inference.multistate import update_split_change  # noqa: E402
from salad_torch.inference.options import (  # noqa: E402
    decode_sequence, parse_dssp, parse_dssp_mean, parse_dssp_split)
from salad_torch.inference.postprocess import reorder_amino_acids  # noqa: E402
from salad_torch.data.dssp import assign_dssp  # noqa: E402
from salad_torch.inference.symmetry import (  # noqa: E402
    compact_clash_update, parse_sym, screw_replicate)
from salad_torch.inference.template import parse_template  # noqa: E402
from salad_torch.io.pdb import snapshot_to_protein, to_pdb  # noqa: E402
from salad_torch.modules.model import StructureDiffusionInference, apply_diffusion  # noqa: E402

#: N036 --f_small/--f_strand: upstream retries a rejected design attempt
#: unboundedly (`while idx < opt.num_designs`, no cap). This port bounds it so
#: an automated run can't hang forever; --f_small/--f_strand default to 1.0
#: (never triggers), so the cap is inert unless a caller opts into a real
#: threshold.
_SYM_FILTER_MAX_RETRIES = 20


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
    ap.add_argument("--batch_size", default="",
                    help="N021. PACKED batching: B designs concatenated along "
                         "the flat residue axis of ONE forward, so each kernel "
                         "does B designs' work in one launch. Composes with "
                         "--group_size (G packed batches per captured graph), "
                         "so one replay advances B*G designs. Needs --fast and "
                         "CUDA. NOT semantically free, unlike --group_size: "
                         "salad's neighbour selection is a global top-k over "
                         "the flat residue axis and its '-1' no-neighbour "
                         "sentinel behaves differently depending on a design's "
                         "offset in that axis (notes/04_PARITY_AND_PERF.md B1, "
                         "an upstream bug this port reproduces), so a packed "
                         "design is NOT the design the same seed produces "
                         "alone. --batch_size 1 (the default) is today's path, "
                         "unchanged. See "
                         ".agent-research/results/N021-packed-batching/.")
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
    # -- N036: salad/training/eval_sym.py's extra flags, see symmetry.py's
    # module docstring for what each one does upstream and why. -----------
    sym.add_argument("--mode", default="screw", choices=["screw", "rotation"],
                     help="Symmetry-operation class (`eval_sym.py`). 'screw' "
                          "(default) is exactly what --sym already "
                          "implements above. 'rotation' selects upstream's "
                          "ScrewVP, which is dead code (raises TypeError on "
                          "step 0 of every invocation upstream, in every "
                          "released version, per an argument-count mismatch "
                          "at its own call site) - passing it here raises "
                          "immediately with that citation instead of "
                          "silently reinterpreting a mode upstream itself "
                          "cannot run.")
    sym.add_argument("--mix_output", default="", choices=["", "couple", "select"],
                     help="Upstream's `eval_sym.py`-only name for the SAME "
                          "couple/select choice --sym_mean already controls "
                          "('couple' = --sym_mean True, anything else = "
                          "False) - that script never had --sym_mean at "
                          "all. '' (default) leaves --sym_mean in charge, "
                          "unchanged from before this flag existed; a "
                          "non-empty value overrides it for hooks A and C "
                          "(hook B is always 'select' regardless, see "
                          "--sym_mean's help).")
    sym.add_argument("--sym_noise", default="True", choices=["True", "False"],
                     help="Whether hook B (the noised-input symmetriser) "
                          "runs at all. 'True' (default) reproduces this "
                          "port's prior, unconditional hook B behaviour "
                          "exactly - the trunk's noised input is forced "
                          "perfectly symmetric every call. 'False' lets "
                          "each subunit copy see independently-sampled "
                          "noise; only the x0 prediction (hook C) still "
                          "gets resynchronised every step.")
    sym.add_argument("--fixcenter_threshold", type=float, default=0.0001,
                     help="`eval_sym.py`'s `do_radius` gate for hooks B and "
                          "C: only re-center the representative subunit and "
                          "re-apply --screw_radius while the CURRENT step's "
                          "SCALED noise level (data['t_pos'] after "
                          "--timescale_pos, not the raw step fraction) is "
                          "above this. The default (0.0001, upstream's own "
                          "CLI default) is a no-op for every schedule this "
                          "port runs end to end; raising it (to roughly the "
                          "scale of your --timescale_pos, e.g. 0.5-5) lets "
                          "late/low-noise refinement drift off the exact "
                          "symmetric placement instead of being snapped "
                          "back to it every step.")
    sym.add_argument("--compact_lr", type=float, default=0.0,
                     help="Learning rate for a per-subunit compacting "
                          "gradient nudge in hook C, right after "
                          "symmetrisation: every atom is pulled toward its "
                          "OWN subunit's mean CA position, scaled by this "
                          "rate and the current step's SCALED noise level. "
                          "0.0 (default) is an exact no-op.")
    sym.add_argument("--clash_lr", type=float, default=0.0,
                     help="Learning rate for an inter-subunit clash-"
                          "repulsion gradient nudge in hook C, alongside "
                          "--compact_lr: CA-CA pairs from different chains "
                          "(or the same chain >16 residues apart) closer "
                          "than 8 A are pushed apart. A DIFFERENT clash "
                          "formula than screw_replicate's own chain_center "
                          "repulsion nudge - see symmetry.py. 0.0 (default) "
                          "is an exact no-op.")
    sym.add_argument("--f_small", type=float, default=1.0,
                     help="Reject and regenerate a design (fresh noise, up "
                          f"to {_SYM_FILTER_MAX_RETRIES} attempts, a "
                          "port-side safety cap - upstream retries "
                          "unboundedly) if its ALA+GLY fraction exceeds "
                          "this at ANY requested out-step. 1.0 (default, "
                          "matching upstream) never triggers.")
    sym.add_argument("--f_strand", type=float, default=1.0,
                     help="Reject and regenerate a design if its "
                          "beta-strand DSSP fraction (assessed on one "
                          "subunit) exceeds this at any out-step. 1.0 "
                          "(default, matching upstream) never triggers.")

    # -- §5.4 / §0: dssp, merge_chains ---------------------------------
    ap.add_argument("--dssp", default="none",
                    help="'none', 'random', a literal 'LHE_' string, or "
                         "'parent/child_1/child_2' for multi-state design "
                         "(Lisanza et al. 2024) - three 'LHE_' strings "
                         "separated by '/', uppercase in the PARENT marking "
                         "residues constrained to the same TERTIARY "
                         "structure in the matching child. A literal "
                         "string's length OVERRIDES --num_aa and discards "
                         "--merge_chains/cyclic num_aa specs (§0); the "
                         "multi-state form additionally runs through its "
                         "own generation loop (run_multistate) and rejects "
                         "--template/--sym/--order_agnostic/--merge_chains/"
                         "--group_size/--batch_size/--fast/--renoise/"
                         "--sample_final - see notes/specs/07_multistate_dssp.md.")
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
# N036: --f_small / --f_strand (symmetry.py's module docstring has the rest)
# ---------------------------------------------------------------------------
def sym_filter_check(snaps: list[dict], replicate_count: int,
                     f_small: float, f_strand: float) -> tuple[bool, str]:
    """`eval_sym.py:392-394,384-390,531-539`'s design-quality gate.

    Checked over every requested out-step snapshot IN ORDER, matching
    upstream's per-out-step `break` inside its design loop: the first
    snapshot that fails either check fails the WHOLE attempt - upstream
    never gives partial credit for cleanly reaching some but not all
    requested out-steps. Returns ``(True, "")`` iff every snapshot passes
    both checks; the reject-and-retry itself lives in `main`'s per-design
    loop, not here.
    """
    for snap in snaps:
        seq = decode_sequence(snap["aatype"])
        alagly = (seq.count("A") + seq.count("G")) / max(len(seq), 1)
        if alagly > f_small:
            return False, f"ALA/GLY {100 * alagly:.1f}% > f_small {100 * f_small:.1f}%"
        if f_strand < 1.0:
            pos = snap["pos"]
            length = pos.shape[0] // replicate_count if replicate_count > 1 else pos.shape[0]
            sub = pos[:length]
            batch_index = torch.zeros((length,), dtype=torch.long, device=sub.device)
            mask = torch.ones((length,), dtype=torch.bool, device=sub.device)
            dssp, _, _ = assign_dssp(sub, batch_index, mask)
            strand_frac = float((dssp == 2).float().mean())
            if strand_frac > f_strand:
                return False, f"strand {100 * strand_frac:.1f}% > f_strand {100 * f_strand:.1f}%"
    return True, ""


# ---------------------------------------------------------------------------
# multi-state design: --dssp "parent/child_1/child_2" (Lisanza et al. 2024)
# ---------------------------------------------------------------------------
#: Flags this loop shape cannot support or that upstream's own
#: `eval_split_change.py` never wires (see `run_multistate`'s docstring for
#: the per-flag reasoning). Checked by attribute name against its "inactive"
#: default/sentinel value.
_MULTISTATE_INCOMPATIBLE = (
    ("template", ""), ("sym", ""), ("order_agnostic", False),
    ("merge_chains", False), ("group_size", ""), ("batch_size", ""),
    ("fast", False), ("renoise", False), ("sample_final", False),
    ("cloud_std", ""),
)


def run_multistate(args: argparse.Namespace, config, model, device: torch.device) -> None:
    """``--dssp "parent/child_1/child_2"``: Lisanza *et al.*, 2024 multi-state
    design. `notes/specs/07_multistate_dssp.md`.

    Three independent single-chain states (parent, child_1, child_2), each
    its own length (from the DSSP string, which - like single-state
    ``--dssp`` - overrides ``--num_aa`` entirely), advanced through the
    ordinary denoising schedule in lockstep via `sample_group` (already
    exactly "N independent designs, one shared clock, per-design state" -
    nothing new needed there). The only genuinely new per-step behavior is
    `update_split_change`, applied to all three states' incoming ``pos``
    *before* each one is noised/denoised - reusing `sample_group`'s
    `step_fn_group` seam, the same seam `run_grouped` already uses for its
    own multi-design batching.

    Secondary-structure conditioning (each state's own `dssp_condition`, from
    its own case-folded substring) and the hard tertiary-structure coupling
    (`same_mask`, parent-case-only) are two independent mechanisms - see
    `options.py::MultiStateDSSP` and `multistate.py::update_split_change`.

    **Incompatible flags, and why** (`_MULTISTATE_INCOMPATIBLE`, checked
    below and rejected with `SystemExit` rather than silently ignored or
    silently mis-combined - matching this port's existing convention for
    flag combinations that would produce confusing, not just "unusual",
    output, e.g. `run_grouped`'s own compatibility checks):

    * `--template`, `--order_agnostic`, `--group_size`, `--batch_size`,
      `--fast`, `--renoise`, `--sample_final`: **not in upstream's own
      `eval_split_change.py` CLI at all** (its `parse_options` call lists
      only `params, out_path, config, timescale_pos, timescale_seq, num_aa,
      sym, merge_chains, variants, sym_mean, num_designs, num_steps,
      out_steps, dssp, prev_threshold, cloud_std, sym_threshold, jax_seed`).
      These are all port-side additions to the *single-state* path (spec
      06); genuinely combining any of them with three coupled,
      independently-lengthed trajectories is new design work this task does
      not ask for (a 3-state renoise loop, a 3-state compiled/CUDA-graph
      path, template conditioning applied to which of the three states,
      etc.) - rejected outright rather than half-implemented.
    * `--sym`, `--merge_chains`, `--cloud_std`: **are** in upstream's
      `eval_split_change.py` CLI (`parse_options` declares `sym`,
      `merge_chains`, `cloud_std` defaults), but the script's actual body
      never reads `opt.sym`/`opt.merge_chains`/`opt.cloud_std` anywhere -
      copy-pasted CLI surface from the shared de-novo script, left dead.
      Upstream's true behavior is "silently do nothing"; this port instead
      rejects them explicitly if passed non-default. A user who typed
      `--merge_chains` expecting *something* to change is better served by a
      clear error than by silent, upstream-faithful inertness that looks
      like a bug in the port.

    `--depth_adjust`/`--compile`/`--seed`/`--jax_seed`/`--sequence_only`
    compose normally (the first two are already applied to `config`/`model`
    before this function is called; the rest are read directly below) and
    are not restricted.
    """
    for name, inactive in _MULTISTATE_INCOMPATIBLE:
        value = getattr(args, name)
        if value != inactive:
            raise SystemExit(
                f"multi-state --dssp 'parent/child_1/child_2' is "
                f"incompatible with --{name}: see run_multistate's "
                f"docstring in scripts/generate.py for why. Drop --{name} "
                f"to run multi-state design.")

    spec = parse_dssp_split(args.dssp.strip())
    lengths = [int(spec.parent.shape[0]), int(spec.child_1.shape[0]),
              int(spec.child_2.shape[0])]
    names = ("parent", "child_1", "child_2")
    dssp_conditions = [spec.parent, spec.child_1, spec.child_2]
    same_mask = spec.same_mask.to(device)

    num_steps = args.num_steps if args.num_steps is not None else 500
    out_steps_s = args.out_steps if args.out_steps is not None else "400"
    cfg = SamplerConfig(num_steps=num_steps,
                        out_steps=tuple(int(x) for x in out_steps_s.split(",")),
                        prev_threshold=args.prev_threshold,
                        timescale_pos=args.timescale_pos)
    print(f"multi-state design: parent={lengths[0]}aa child_1={lengths[1]}aa "
          f"child_2={lengths[2]}aa (parent_split={spec.parent_split}, "
          f"{int(same_mask.sum())} tertiary-coupled residues)  "
          f"steps={num_steps} out_steps={cfg.out_steps}")

    os.makedirs(args.out_path, exist_ok=True)
    seq_path = os.path.join(args.out_path, "sequences.fa")

    def step_fn_group(ds, ps, gs, *, need_aa=True):
        del need_aa   # every step needs the full output here (no lean path)
        with torch.no_grad():
            ds[0]["pos"], ds[1]["pos"], ds[2]["pos"] = update_split_change(
                ds[0]["pos"], ds[1]["pos"], ds[2]["pos"],
                same_mask, spec.parent_split)
            updates, new_prevs = [], []
            for d, p, g in zip(ds, ps, gs):
                noised = apply_diffusion(config, d, generator=g)
                fed = dict(d, **noised)
                u, new_p = model.predict(fed, p)
                updates.append(u)
                new_prevs.append(new_p)
            return updates, new_prevs

    times = []
    for design in range(args.num_designs):
        datas, prevs, gens = [], [], []
        for i, length in enumerate(lengths):
            resi = torch.arange(length, dtype=torch.long)
            chain = torch.zeros(length, dtype=torch.long)
            data, prev = make_initial_data(length, resi, chain, None, device=device)
            data["dssp_condition"] = dssp_conditions[i].to(device)
            datas.append(data)
            prevs.append(prev)
            # one independent, seed-derived stream per (design, state) - see
            # design_generator's docstring for why sequential designs cannot
            # share one generator once anything runs in lockstep.
            gens.append(torch.Generator(device=device).manual_seed(
                (args.seed * 1_000_003 + design * 3 + i) & 0x7FFF_FFFF_FFFF_FFFF))

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        snaps_per_state = sample_group(step_fn_group, datas, prevs, cfg,
                                       generators=gens)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        for i, name in enumerate(names):
            for snap in snaps_per_state[i]:
                step = snap["step"]
                res = {k: (v.cpu() if torch.is_tensor(v) else v)
                      for k, v in snap.items()}
                if args.sequence_only:
                    seq = decode_sequence(res["aatype"])
                    with open(seq_path, "at") as fh:
                        fh.write(f">design_{design}_{name}_{step}\n{seq}\n")
                        fh.flush()
                else:
                    path = os.path.join(args.out_path,
                                        f"design_{design}_{name}_{step}.pdb")
                    write_result_pdb(path, res,
                                     chain_index=torch.zeros(lengths[i], dtype=torch.long),
                                     order_agnostic=False)
        print(f"  design {design}: {elapsed:6.1f}s  {len(snaps_per_state[0])} "
              f"out-step(s)  -> {args.out_path}")

    t = np.array(times)
    print(f"\n{args.num_designs} multi-state designs "
          f"(parent {lengths[0]}aa / child_1 {lengths[1]}aa / "
          f"child_2 {lengths[2]}aa): median {np.median(t):.1f}s, "
          f"total {t.sum():.1f}s")
    if args.timing_json:
        with open(args.timing_json, "w") as fh:
            json.dump(dict(backend="torch", config=args.config,
                           num_aa=lengths, num_steps=cfg.num_steps,
                           device=str(device), multistate=True,
                           seconds=times), fh, indent=2)


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
    """`--group_size` (N019) and/or `--batch_size` (N021).

    Two orthogonal axes of parallelism over designs, and they compose:

    * ``--group_size G`` records G INDEPENDENT copies of the forward into one
      CUDA graph on G streams. Nothing is shared, so design `i` computes
      exactly what it would alone - measured bit-identical under
      ``SALAD_OPT_DETSEG=1``. It reclaims idle SMs; it cannot make a kernel
      bigger.
    * ``--batch_size B`` concatenates B designs along salad's flat residue
      axis, so every kernel does B designs' work in ONE launch. It makes the
      kernels bigger - and it is NOT semantically free: the neighbour
      selection is a global top-k over that axis, so a design's answer
      depends on its offset in it (`notes/04_PARITY_AND_PERF.md` B1).

    One replay therefore advances ``B * G`` designs. The design-facing
    behaviour is the ordinary one - ``design_{n}.pdb``, the same stats line,
    the same ``--timing_json``.

    A trailing group is run at a smaller G rather than padded with throwaway
    designs. A trailing *pack* (``--num_designs`` not a multiple of B) is run
    by a second, smaller-B denoiser, which costs one extra compile and is
    announced; padding it would be the one thing B1 forbids.
    """
    from salad_torch.modules.fast import GroupFastDenoiser

    if not args.fast:
        raise SystemExit("--group_size/--batch_size need --fast (they group or "
                         "pack the captured graph; there is nothing to group "
                         "on the eager path)")
    if device.type != "cuda":
        raise SystemExit("--group_size/--batch_size need a CUDA device")
    for flag in ("sym", "renoise", "sequence_only", "sample_final"):
        if getattr(args, flag) not in ("", False, None):
            raise SystemExit(
                f"--group_size/--batch_size are incompatible with --{flag}: "
                f"they step every design through one captured replay, and "
                f"--{flag} inserts per-design host work between the noising "
                f"and the denoising that the capture cannot see. Drop them "
                f"to get the ordinary path back.")

    want = (args.group_size or "1").strip().lower()
    if want != "auto":
        try:
            want = int(want)
        except ValueError:
            raise SystemExit(f"--group_size must be 'auto' or an integer, "
                             f"got {args.group_size!r}")
        if want < 1:
            raise SystemExit("--group_size must be >= 1")
    try:
        batch = int((args.batch_size or "1").strip())
    except ValueError:
        raise SystemExit(f"--batch_size must be an integer, "
                         f"got {args.batch_size!r}")
    if batch < 1:
        raise SystemExit("--batch_size must be >= 1")

    pc_hint = bool(config.multi_motif or pc["template"] is not None
                   or int(chain.max()) > 0)
    os.makedirs(args.out_path, exist_ok=True)
    stats, times, group_wall = [], [], []
    written = 0
    reported_n = None

    # PHASES. One per distinct batch_size: the bulk of the run at B, then -
    # only if `--num_designs` is not a multiple of B - a tail at B_tail = the
    # remainder. A second phase is a second compile, so it is announced.
    tail = args.num_designs % batch
    bulk = args.num_designs - tail
    phases = ([(0, bulk, batch, want)] if bulk else [])
    if tail:
        phases.append((bulk, tail, tail, 1))

    for first, count, b, group_request in phases:
        if first:
            print(f"  --num_designs {args.num_designs} is not a multiple of "
                  f"--batch_size {batch}; the trailing {tail} design(s) need "
                  f"their own compile at batch_size={tail} (padding the pack "
                  f"is what notes/04_PARITY_AND_PERF.md B1 forbids). Pick a "
                  f"multiple of {batch} to avoid this.")
        t_warm = time.perf_counter()
        group = GroupFastDenoiser(model.predict, num_aa,
                                  group_size=group_request, device=device,
                                  preset=args.preset,
                                  batch_size=b).warmup(pair_cond=pc_hint)
        n = group.group_size
        per_replay = group.designs_per_replay
        if reported_n is None:
            reported_n = per_replay
        peak_gb = (group.peak_bytes or 0) / 2 ** 30
        per_gb = (group.per_member_bytes or 0) / 2 ** 30
        if group_request == "auto":
            props = torch.cuda.get_device_properties(device)
            print(f"group_size auto -> {n} on {props.name} "
                  f"({props.multi_processor_count} SMs, "
                  f"{props.total_memory / 2**30:.1f} GiB) at batch_size={b}")
        print(f"grouped fast path ({args.preset}, batch_size={b}, "
              f"group_size={n} -> {per_replay} designs/replay) compiled + "
              f"captured in {time.perf_counter() - t_warm:.0f}s (one-time); "
              f"peak reserved {peak_gb:.2f} GiB, "
              f"~{per_gb:.3f} GiB/packed member")
        if n == 1 and b == 1:
            print("  batch_size 1 + group_size 1: this is the ordinary "
                  "single-design computation, run through the grouped "
                  "machinery - the reference the other configurations must "
                  "be read against.")
        if b > 1:
            print(f"  NB batch_size {b} is NOT bit-identical to batch_size 1: "
                  f"a design's answer depends on its offset in the flat "
                  f"residue axis (upstream bug B1). Same distribution, "
                  f"different designs.")

        for start in range(first, first + count, per_replay):
            members = min(n, -(-(first + count - start) // b))
            design_ids = list(range(start, min(start + members * b,
                                               first + count)))
            if members != n:
                # trailing GROUP: recapture at fewer members. One extra
                # capture, not one extra compile - the compiled forward and
                # every member's buffers survive.
                group.resize(members, pair_cond=pc_hint)
                n = group.group_size
                per_replay = group.designs_per_replay
            group.invalidate_condition_cache()
            datas, prevs, gens, lengths = [], [], [], []
            for m in range(members):
                ids = design_ids[m * b:(m + 1) * b]
                per_design = [make_design_data(args, pc, num_aa, resi, chain,
                                               is_cyclic, cyclic_mask, device)
                              for _ in ids]
                pdata, pprev, lens = pack_designs([d for d, _ in per_design],
                                                  [p for _, p in per_design])
                datas.append(pdata)
                prevs.append(pprev)
                lengths.append(lens)
                # keyed on the FIRST design of the pack, so batch_size=1
                # reproduces N019's per-design streams exactly.
                gens.append(design_generator(device, args.seed, ids[0]))
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
            snaps_per_member = sample_group(step_fn_group, datas, prevs, cfg,
                                            generators=gens)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            n_designs = len(design_ids)
            # one entry PER DESIGN, not per replay - the summary line below
            # sums this to get the run's total GPU time.
            times.extend([elapsed / n_designs] * n_designs)
            group_wall.append(elapsed)

            flat_ids = iter(design_ids)
            for m, snaps in enumerate(snaps_per_member):
                for part in split_packed_snapshot(snaps[-1], lengths[m]):
                    d = next(flat_ids)
                    snap = {k: (v.cpu() if torch.is_tensor(v) else v)
                            for k, v in part.items() if k != "result"}
                    path = os.path.join(args.out_path, f"design_{d}.pdb")
                    write_result_pdb(path, snap, chain_index=base_chain,
                                     order_agnostic=args.order_agnostic)
                    ca = snap["pos"][:, 1].numpy()
                    d_ca = np.linalg.norm(ca[1:] - ca[:-1], axis=-1)
                    rg = float(np.sqrt(((ca - ca.mean(0)) ** 2).sum(-1).mean()))
                    stats.append(dict(ca_ca_mean=float(d_ca.mean()),
                                      ca_ca_min=float(d_ca.min()),
                                      ca_ca_max=float(d_ca.max()), rg=rg,
                                      seconds=elapsed / n_designs))
                    written += 1
                    print(f"  design {d}: {elapsed / n_designs:6.1f}s/design  "
                          f"CA-CA {d_ca.mean():.3f} A (min {d_ca.min():.3f} "
                          f"max {d_ca.max():.3f})  Rg {rg:.1f} A  -> {path}")
            print(f"  replay of {n_designs} design(s) "
                  f"({members} x {b}): {elapsed:.1f}s wall "
                  f"({elapsed / n_designs:.3f}s/design)")
        del group
        torch.cuda.empty_cache()

    t = np.array(times)
    print(f"\n{written} designs of {num_aa} aa, up to {reported_n} per "
          f"replay (batch_size={batch}): "
          f"median {np.median(t):.3f}s/design, "
          f"{sum(group_wall):.1f}s of GPU time for all {written} "
          f"(compile/capture excluded)")
    if args.timing_json:
        with open(args.timing_json, "w") as fh:
            json.dump(dict(backend="torch", config=args.config, num_aa=num_aa,
                           num_steps=args.num_steps, device=str(device),
                           compile=args.compile,
                           batch_size=batch, designs_per_replay=reported_n,
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

    # multi-state design (Lisanza et al. 2024): a literal --dssp value
    # containing "/" is "parent/child_1/child_2", not a single-state string,
    # and needs its own generation loop (three coupled, independently-lengthed
    # trajectories) - dispatch before apply_precedence_chain's single-state
    # --dssp handling, which would otherwise choke on the "/" characters.
    if "/" in args.dssp.strip():
        run_multistate(args, config, model, device)
        return

    pc = apply_precedence_chain(args)
    num_aa = pc["num_aa"]
    resi, chain, base_chain = pc["resi"], pc["chain"], pc["base_chain"]
    is_cyclic, cyclic_mask = pc["is_cyclic"], pc["cyclic_mask"]

    sym = parse_sym(args.sym, replicate_count=args.replicate_count,
                    screw_angle=args.screw_angle, screw_radius=args.screw_radius,
                    screw_translation=args.screw_translation,
                    sym_mean=args.sym_mean, sym_threshold=args.sym_threshold,
                    mode=args.mode, mix_output=args.mix_output,
                    sym_noise=args.sym_noise,
                    fixcenter_threshold=args.fixcenter_threshold,
                    compact_lr=args.compact_lr, clash_lr=args.clash_lr,
                    f_small=args.f_small, f_strand=args.f_strand)
    # N036: the timescale function `sample()` uses internally, re-parsed here
    # (pure/cached, semantically free - see `parse_timescale`'s docstring) so
    # hook A/C can recompute the CURRENT step's *scaled* noise level from
    # `raw_t` for --fixcenter_threshold/--compact_lr/--clash_lr, which key off
    # upstream's `data["t_pos"][0]` (scaled), not the raw step fraction
    # --sym_threshold already uses.
    timescale_pos_fn = parse_timescale(args.timescale_pos)

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

    if args.group_size or args.batch_size:
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
            # N036: --fixcenter_threshold's do_radius at raw_t=1.0 (the
            # trajectory's start, which is what hook A stands in for -
            # upstream's eval_sym.py has no separate "once, before the loop"
            # pass at all; the very first per-step call plays this role
            # there, on the all-zero initial cloud).
            do_radius0 = timescale_pos_fn(1.0) > sym.fixcenter_threshold
            data["pos"] = screw_replicate(
                data["pos"], count=sym.replicate_count, angle=sym.angle_rad,
                radius=sym.radius, translation=sym.translation,
                chain_center=False, mean=sym.mean, do_radius=do_radius0)

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
                # N036 --sym_noise: upstream's `config.replicate_noise` gate
                # ("True" is this port's prior, unconditional behaviour).
                # When off, skip the call outright - each subunit copy keeps
                # apply_diffusion's independent per-residue noise.
                if _sym.sym_noise:
                    # N036 --fixcenter_threshold: upstream reads this off
                    # `data["t_pos"][0]`, the CURRENT step's scaled noise
                    # level, already sitting in `d` at this point (sample()
                    # sets it before calling step_fn).
                    scaled_t = float(d["t_pos"].reshape(-1)[0])
                    do_radius = scaled_t > _sym.fixcenter_threshold
                    d["pos_noised"] = screw_replicate(
                        d["pos_noised"], count=_sym.replicate_count,
                        angle=_sym.angle_rad, radius=_sym.radius,
                        translation=_sym.translation, chain_center=False,
                        mean=False, do_radius=do_radius)
                return model.predict(d, p)

        def post_step_fn(pos, raw_t, _sym=sym, _tsfn=timescale_pos_fn,
                        _chain=data["chain_index"], _resi=data["residue_index"]):
            # Hook C (§1.6): post-hoc per-step symmetrisation of the x0
            # prediction, gated by --sym_threshold. The repulsion nudge
            # (chain_center) only fires for raw_t > 0.5 and scales with raw_t.
            if not _sym.active or raw_t <= _sym.threshold:
                return pos
            # N036 --fixcenter_threshold: post_step_fn only gets raw_t, so
            # recompute the SAME scaled value sample() already set on
            # data["t_pos"] for this step (pure function of raw_t - see
            # timescale_pos_fn's definition above).
            scaled_t = _tsfn(raw_t)
            do_radius = scaled_t > _sym.fixcenter_threshold
            pos = screw_replicate(
                pos, count=_sym.replicate_count, angle=_sym.angle_rad,
                radius=_sym.radius, translation=_sym.translation,
                chain_center=(raw_t > 0.5), time=raw_t, mean=_sym.mean,
                do_radius=do_radius)
            # N036 --compact_lr/--clash_lr: additive guidance nudges on
            # the just-symmetrised x0, applied right here in eval_sym.py
            # (`model_step`'s steps 2-4, before the NEXT step's noising -
            # exactly where this runs too, since whatever post_step_fn
            # leaves in data["pos"] is what the next step_fn call noises).
            if _sym.compact_lr or _sym.clash_lr:
                pos = compact_clash_update(
                    pos, replicate_count=_sym.replicate_count,
                    chain_index=_chain, residue_index=_resi,
                    compact_lr=_sym.compact_lr, clash_lr=_sym.clash_lr,
                    t=scaled_t)
            return pos

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
        # -- N036 --f_small/--f_strand: reject-and-regenerate ----------------
        # `eval_sym.py`'s design loop discards and redraws (fresh noise, same
        # design index - upstream never increments `idx` on failure) a whole
        # attempt if ANY requested out-step fails either check; no partial
        # credit. Both default to 1.0 (never triggers), so for every command
        # line that doesn't opt in this loop's body runs 0 times - `gen`
        # (the seeded generator) has already advanced past this design's
        # steps either way, exactly like upstream's continuously-advancing
        # `key`.
        sym_retries = 0
        while sym.active and (sym.f_small < 1.0 or sym.f_strand < 1.0):
            ok, reason = sym_filter_check(snaps, sym.replicate_count,
                                          sym.f_small, sym.f_strand)
            if ok:
                break
            sym_retries += 1
            if sym_retries > _SYM_FILTER_MAX_RETRIES:
                print(f"  design {design}: giving up after {sym_retries - 1} "
                      f"rejected attempt(s) ({reason}); keeping this one "
                      f"anyway (port-side safety cap, upstream retries "
                      f"unboundedly)")
                break
            print(f"  design {design}: rejected ({reason}), regenerating "
                  f"(attempt {sym_retries + 1})")
            if fast_denoiser is not None:
                fast_denoiser.invalidate_condition_cache()
            data, prev = make_design_data(args, pc, num_aa, resi, chain,
                                          is_cyclic, cyclic_mask, device)
            do_radius0 = timescale_pos_fn(1.0) > sym.fixcenter_threshold
            data["pos"] = screw_replicate(
                data["pos"], count=sym.replicate_count, angle=sym.angle_rad,
                radius=sym.radius, translation=sym.translation,
                chain_center=False, mean=sym.mean, do_radius=do_radius0)
            snaps = sample(step_fn, data, prev, cfg, generator=gen,
                           post_step_fn=post_step_fn, final_step_fn=final_step_fn)
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
