"""Veto the N017/O1 inductor CSE pass without breaking torch's own imports.

At fd9ba32 the pass is a LOCAL CLOSURE inside `FastDenoiser.__init__`:

    from torch._functorch.compile_utils import fx_graph_cse as _cse
    def _cse_pass(graph):
        graph.owning_module.graph = _cse(graph)
    _ind.joint_custom_post_pass = _cse_pass

so it cannot be nulled from outside after construction, and `joint_custom_post_pass`
is re-set every time a FastDenoiser is built. DELETING `fx_graph_cse` does reach the
module's own `except ImportError` branch, but it also breaks
`torch._functorch.partitioners`, which does a module-level
`from .compile_utils import fx_graph_cse` - so the whole inductor import fails.

The veto that works: let the import succeed, but make the pass the IDENTITY.
`_cse_pass` then assigns the graph back to itself, which is a no-op. torch's own
module-level import is taken first so the partitioner keeps the real function.
"""


def veto_cse():
    import torch._functorch.partitioners  # binds the REAL fx_graph_cse first
    import torch._functorch.compile_utils as cu

    real = cu.fx_graph_cse
    cu.fx_graph_cse = lambda g: g
    assert torch._functorch.partitioners.fx_graph_cse is real
    print("[n025] inductor CSE pass vetoed (fx_graph_cse -> identity)", flush=True)
