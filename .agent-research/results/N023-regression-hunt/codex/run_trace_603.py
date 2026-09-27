"""Read-only sampling trace: record C-alpha after each complete denoiser step."""
import os
import sys
sys.path.insert(0, "/home/hpc/v181bb/v181bb10/SALAD_torch/.agent-research/results/N023-regression-hunt/codex/603")
import numpy as np
from scripts import generate as gen

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
