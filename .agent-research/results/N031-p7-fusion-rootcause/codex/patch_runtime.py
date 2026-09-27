"""Test-only runtime application of the proposed one-line forward_core patch."""
import inspect,textwrap,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))

def install():
    import salad_torch.modules.diffusion as md
    import salad_torch.modules.fast as fast
    old=md.Diffusion.forward_core
    source=textwrap.dedent(inspect.getsource(old))
    needle='''        cond_nbr_stack = torch.stack(
            [cond_neighbours_of(i, prev_pos) for i in range(self.depth)])
        cond_pair = self.diffusion_stack.cond_pair_stack('''
    replacement='''        cond_nbr_stack = torch.stack(
            [cond_neighbours_of(i, prev_pos) for i in range(self.depth)])
        torch._dynamo.graph_break()  # N026: materialize before relative-position gather
        cond_pair = self.diffusion_stack.cond_pair_stack('''
    if source.count(needle)!=1:
        raise RuntimeError(f'expected one P7 site, found {source.count(needle)}')
    namespace=dict(md.__dict__)
    exec(compile(source.replace(needle,replacement),old.__code__.co_filename,'exec'),namespace)
    md.Diffusion.forward_core=namespace['forward_core']
    fast.PRESETS['fast']['batch_cond_pair']=True
    fast.PRESETS['exact']['batch_cond_pair']=True
    if os.getenv('N026_REQUIRE_P7_EAGER') == '1':
        from salad_torch.modules.blocks import DiffusionStack
        old_available = DiffusionStack.batched_cond_pair_available
        def checked_available(self):
            result = old_available(self)
            if self.cond_pair_stack is not None and not result:
                raise RuntimeError('N026 eager gate: eligible stack did not enable P7')
            marker = os.getenv('N026_EAGER_MARKER')
            if result and marker:
                with open(marker, 'a') as f:
                    f.write('P7=True\n')
            return result
        DiffusionStack.batched_cond_pair_available = checked_available
    if os.getenv('N026_REQUIRE_P7') == '1':
        old_init = fast.FastDenoiser.__init__
        def checked_init(self, *args, **kwargs):
            old_init(self, *args, **kwargs)
            if not md.opt.BATCH_COND:
                raise RuntimeError('N026 P7 gate: FastDenoiser did not enable BATCH_COND')
            marker = os.getenv('N026_MARKER')
            if marker:
                with open(marker, 'a') as f:
                    f.write('BATCH_COND=True\n')
        fast.FastDenoiser.__init__ = checked_init
    print('[N026 patch] graph break after cond_nbr_stack; P7 preset enabled',flush=True)
