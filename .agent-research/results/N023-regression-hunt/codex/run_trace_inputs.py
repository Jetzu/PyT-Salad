"""Read-only sampling trace: record C-alpha after each complete denoiser step."""
import os
import sys
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch")
import numpy as np
from scripts import generate as gen
from salad_torch.modules.fast import FastDenoiser
_original_run = FastDenoiser._run
_seen = False
def traced_run(self, inputs, *args, **kwargs):
    global _seen
    if not _seen:
        keep = ("pos_noised", "prev_pos", "prev_local", "condition", "uniform", "t_pos", "chain", "resi")
        np.savez_compressed(os.environ["INPUT_OUT"], **{k: inputs[k].detach().cpu().numpy().copy() for k in keep})
        _seen = True
    return _original_run(self, inputs, *args, **kwargs)
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
