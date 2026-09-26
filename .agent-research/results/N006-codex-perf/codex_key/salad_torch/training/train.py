"""CLI entry point for `default_vp` training - replaces
`train_noise_schedule_benchmark.py`.

Two datasources are wired up (spec §A.4-A.5, §0d):

* ``--synthetic`` (default, and the only one exercised on this machine): an
  in-memory, idealised-geometry source (`data.dataset.SyntheticStructureSource`)
  - no preprocessed PDB caches needed, so this can run today.
* ``--data-path PATH``: the real `NpzAssemblySource` (`{PATH}/allpdb/...`,
  spec §A.1-A.2). **Untested** - this checkout has no local `.npz`/cluster-tsv/
  `entries.idx` tree; wire it in once one is reachable (see
  `notes/triad_protein_database_locations.md`-style pointers in the wider repo
  for where such a tree might live on this cluster).

`rebatch`/`multigpu`/`pmap`/Haiku-tabulate/`nanhunt` (spec §B.1, §B.7) are
deliberately not ported - incidental cluster-scaling/debug machinery for a
single-process torch run. `find_unused_parameters=True` is the one DDP
requirement that would NOT be optional if this were wrapped in
`DistributedDataParallel` (spec §B.5) - left to the caller, since this script
targets a single process/device.
"""
from __future__ import annotations

import argparse
import logging

import torch

from ..config import get_config
from ..data.dataset import NpzAssemblySource, PackedProteinStream, SyntheticStructureSource
from ..modules.model import StructureDiffusionPredict
from .loop import TrainConfig, Trainer

__all__ = ["build_argparser", "main"]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="default_vp", choices=["default_vp", "default_vp_scaled",
                                                              "default_ve_scaled", "multimotif_vp"])
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--resume", default=None, help="path to a save.pt to resume from")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    # -- data pipeline (spec §A.4/§B.1) --------------------------------------
    p.add_argument("--synthetic", action="store_true", default=True,
                   help="use the in-memory synthetic source (default; the "
                        "real --data-path source is untested on this machine)")
    p.add_argument("--data-path", default=None,
                   help="root of a real {data_path}/allpdb/ tree (§A.1); "
                        "overrides --synthetic when given")
    p.add_argument("--num-aa", type=int, default=1024, help="residues per draw ('size')")
    p.add_argument("--min-size", type=int, default=16)
    p.add_argument("--p-complex", type=float, default=0.5)
    p.add_argument("--legacy-repetitive", type=lambda s: s != "False", default=True)

    # -- optimizer / schedule (spec §B.1-B.3) --------------------------------
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--decay-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--decay-steps", type=int, default=500_000)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--b1", type=float, default=0.9)
    p.add_argument("--b2", type=float, default=0.99)
    p.add_argument("--ema-weight", type=float, default=0.999)
    p.add_argument("--schedule-fixed", action="store_true", default=False)
    p.add_argument("--accumulate", type=int, default=1)

    # -- checkpointing --------------------------------------------------------
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--save-interval-seconds", type=float, default=600.0)
    p.add_argument("--max-steps", type=int, default=None)

    # -- logging --------------------------------------------------------------
    p.add_argument("--tensorboard-dir", default=None)
    p.add_argument("--log-every", type=int, default=50)
    return p


def build_source(args: argparse.Namespace):
    if args.data_path is not None:
        return NpzAssemblySource(f"{args.data_path}/allpdb/")
    return SyntheticStructureSource(num_clusters=512,
                                    chain_length_range=(16, min(300, args.num_aa)))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = build_argparser().parse_args(argv)
    torch.manual_seed(args.seed)

    model_config = get_config(args.config)
    model = StructureDiffusionPredict(model_config)

    train_config = TrainConfig(
        num_aa=args.num_aa, min_size=args.min_size, p_complex=args.p_complex,
        legacy_repetitive_chains=args.legacy_repetitive, lr=args.lr,
        decay_lr=args.decay_lr, warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps, clip=args.clip, b1=args.b1, b2=args.b2,
        ema_weight=args.ema_weight, schedule_fixed=args.schedule_fixed,
        accumulate=args.accumulate, checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        save_interval_seconds=args.save_interval_seconds,
        seed=args.seed, max_steps=args.max_steps)

    source = build_source(args)
    stream = PackedProteinStream(source, size=train_config.num_aa,
                                 min_size=train_config.min_size,
                                 max_size=train_config.max_size,
                                 p_complex=train_config.p_complex,
                                 legacy_repetitive_chains=train_config.legacy_repetitive_chains,
                                 seed=args.seed)

    tb_writer = None
    if args.tensorboard_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(args.tensorboard_dir)

    trainer = Trainer(model, train_config, iter(stream), device=args.device,
                      tb_writer=tb_writer)
    if args.resume is not None:
        trainer.resume(args.resume)

    def _log(step: int, terms: dict[str, float]) -> None:
        if step % args.log_every == 0:
            logging.info("step %d  total=%.4f  lr=%.2e", step, terms["total"], terms["lr"])

    trainer.run(on_step=_log)


if __name__ == "__main__":
    main()
