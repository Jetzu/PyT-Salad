"""Is the JAX reference number a TF32 number? XLA:GPU defaults to TF32 for fp32 dots."""
import pickle, time
import numpy as np, jax, jax.numpy as jnp, haiku as hk
import salad.modules.noise_schedule_benchmark as NSB
from salad.modules.config import noise_schedule_benchmark as configs
cfg = getattr(configs,"default_vp"); cfg.eval=True
params = pickle.load(open("/hnvme/workspace/v181bb10-salad/params/default_vp-200k.jax","rb"))
L=100; A=5+cfg.augment_size
rng=np.random.default_rng(0); z=np.zeros(L,np.int32)
t=np.arange(L)[:,None]
bb=np.concatenate([2.3*np.cos(t*1.7),2.3*np.sin(t*1.7),1.5*t],-1)
pos=(bb[:,None,:]+rng.normal(0,1.2,(L,A,3))).astype(np.float32)
data=dict(pos=pos,pos_noised=pos.copy(),aa_gt=np.full(L,20,np.int32),
  seq=np.full(L,20,np.int32),seq_noised=np.full(L,20,np.int32),
  residue_index=np.arange(L,dtype=np.int32),chain_index=z,batch_index=z,
  mask=np.ones(L,bool),cyclic_mask=np.zeros(L,bool),
  t_pos=np.full(L,0.5,np.float32),t_seq=np.ones(L,np.float32))
prev=dict(pos=np.zeros_like(pos),local=np.zeros((L,128),np.float32))
data=jax.tree_util.tree_map(jnp.asarray,data); prev=jax.tree_util.tree_map(jnp.asarray,prev)
def fn(d,p): return NSB.StructureDiffusionPredict(cfg)(d,p)
key=jax.random.PRNGKey(0)
for prec in (None,"highest"):
    if prec: jax.config.update("jax_default_matmul_precision", prec)
    model=jax.jit(hk.transform(fn).apply)
    out=model(params,key,data,prev); jax.block_until_ready(out)
    ts=[]
    for _ in range(20):
        t0=time.perf_counter(); out=model(params,key,data,prev); jax.block_until_ready(out)
        ts.append(time.perf_counter()-t0)
    print(f"JAX precision={str(prec):<8} {np.median(ts)*1e3:6.2f} ms   500-step {np.median(ts)*500:5.2f} s")
