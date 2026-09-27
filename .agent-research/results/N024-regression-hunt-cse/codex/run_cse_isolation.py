"""Run production generator with CSE toggled before FastDenoiser builds."""
import runpy
import sys
import os
sys.path.insert(0, os.getcwd())
arm = sys.argv.pop(1)
if arm == "off":
    import salad_torch.modules.fast as fast
    import torch._inductor.config as ind
    fast._cse_graph_pass = None
    ind.joint_custom_post_pass = None
elif arm == "tf32off":
    import salad_torch.modules.fast as fast
    fast.PRESETS["fast"]["tf32"] = False
elif arm != "on":
    raise ValueError(arm)
sys.argv[0] = "scripts/generate.py"
runpy.run_path("scripts/generate.py", run_name="__main__")
