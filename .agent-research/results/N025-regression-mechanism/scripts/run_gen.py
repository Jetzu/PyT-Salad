"""Run scripts/generate.py from a given worktree, optionally with the N017/O1
inductor CSE pass vetoed.

The pass is installed as a LOCAL CLOSURE inside FastDenoiser.__init__ at
fd9ba32, so it cannot be nulled from outside after the fact. The supported
veto is the one the code already has: make the private torch import fail, which
falls into the module's own `except (ImportError, AttributeError): pass`.
"""
import os, runpy, sys

root = sys.argv[1]
argv = sys.argv[2:]
sys.path.insert(0, root)
if os.environ.get("N025_NOCSE") == "1":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from n025_veto import veto_cse
    veto_cse()
sys.argv = ["generate.py"] + argv
runpy.run_path(os.path.join(root, "scripts", "generate.py"), run_name="__main__")
