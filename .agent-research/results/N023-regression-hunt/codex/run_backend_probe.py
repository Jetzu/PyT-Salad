"""Read-only sampling trace: record C-alpha after each complete denoiser step."""
import os
import sys
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch")
import numpy as np
import torch
from scripts import generate as gen
from salad_torch.modules.fast import FastDenoiser
_original_run = FastDenoiser._run
_seen = False
def traced_run(self, inputs, need_aa=True, pair_cond=True):
    global _seen
    out = _original_run(self, inputs, need_aa=need_aa, pair_cond=pair_cond)
    if not _seen:
        from salad_torch.modules.fast import _CORE_ARGS
        args = [inputs.get(k) for k in _CORE_ARGS] + [self.need_aux, need_aa, pair_cond]
        with torch.no_grad():
            eager = self.model.diffusion.forward_core(*args)
            compiled = self._compiled(*args)
        def diag(label, x, y):
            d = float((x[0]-y[0]).abs().max())
            def rg(z):
                ca=z[0][:,1,:]
                return float(((ca-ca.mean(0)).square().sum(-1).mean()).sqrt())
            print('BACKEND', label, 'max_abs', d, 'rg_x', rg(x), 'rg_y', rg(y), flush=True)
        diag('graph_vs_eager', out, eager)
        diag('compiled_vs_eager', compiled, eager)
        diag('graph_vs_compiled', out, compiled)
        _seen = True
    return out
FastDenoiser._run = traced_run

_original_sample = gen.sample

def traced_sample(step_fn, data, prev, cfg, **kwargs):
    frames = []
    def traced_step(*args, **kw):
        update, new_prev = step_fn(*args, **kw)
        frames.append(update['pos'][:, 1, :].detach().cpu().numpy().copy())
        return update, new_prev
    result = _original_sample(traced_step, data, prev, cfg, **kwargs)
    np.savez_compressed(os.environ['TRACE_OUT'], ca=np.stack(frames))
    print('TRACE_STEPS', len(frames), 'CA_SHAPE', frames[0].shape)
    return result

gen.sample = traced_sample
gen.main()
