"""Run `scripts/generate.py` with the trunk's neighbourhood halved, K 64 -> 32.

An EXPERIMENT, not a shipped path: it monkeypatches the neighbour-stage counts
before `generate.py` builds the model, so no production file has to carry a
switch for a change that may not survive the quality gate.

  current branch  extract_neighbours(16, 16, 32) -> (16, 8, 8)
  condition branch  spatial 16 -> 8, random 16 -> 8

Every argument after `--` is passed straight to `generate.py`.

  python x_gen_k32.py -- --num_aa 100 --num_designs 8 --fast --preset fast \
                         --out_path <dir>
"""
from __future__ import annotations

import runpy, sys

ROOT = "/home/hpc/v181bb/v181bb10/SALAD_torch"
sys.path.insert(0, ROOT)

import salad_torch.modules.diffusion as DF   # noqa: E402

_orig_cur = DF.extract_neighbours
_orig_pro = DF.condition_neighbour_prologue
_orig_tail = DF.condition_neighbour_tail
_orig_ecn = DF.extract_condition_neighbours

# The originals are bound as DEFAULT ARGUMENTS, not read from globals:
# `runpy.run_path(run_name="__main__")` below replaces this module's globals
# with generate.py's, so a global lookup inside the lambda resolves against
# the WRONG module at trace time and Dynamo aborts with
# `module '__main__' has no attribute '_orig_cur'`.
DF.extract_neighbours = lambda a, b, c, _f=_orig_cur: _f(16, 8, 8)
DF.condition_neighbour_prologue = lambda a, b, _f=_orig_pro: _f(a, 8)
DF.condition_neighbour_tail = lambda a, b, _f=_orig_tail: _f(8, b)
DF.extract_condition_neighbours = lambda a, b, c, d, _f=_orig_ecn: _f(a, 8, 8, d)

argv = sys.argv[1:]
if argv and argv[0] == "--":
    argv = argv[1:]
sys.argv = [f"{ROOT}/scripts/generate.py"] + argv
print("[x_gen_k32] neighbourhood halved: current (16,8,8), condition (16,8,8)")
runpy.run_path(f"{ROOT}/scripts/generate.py", run_name="__main__")
