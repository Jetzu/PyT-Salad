import argparse
from salad_torch.training.train import build_source
from salad_torch.data.dataset import PackedProteinStream
for n in (16,32):
    try:
        source=build_source(argparse.Namespace(data_path=None,num_aa=n))
        item=PackedProteinStream(source,size=n,min_size=16,max_size=n,seed=7,max_attempts_per_draw=1000).next_item()
        print('DATA',n,'valid',int(item['mask'].sum()),'shape',item['all_atom_positions'].shape)
    except Exception as e: print('DATA',n,type(e).__name__,str(e)[:300])
