import sys
import numpy as np, jax, jax.numpy as jnp, haiku as hk
import salad.modules.noise_schedule_benchmark as NSB
import salad.modules.utils.geometry as G
from salad.aflib.model.geometry import Vec3Array
from salad.modules.config import noise_schedule_benchmark as configs
D=np.load(sys.argv[1]); NPZ=np.load(sys.argv[2])
U=jnp.asarray(D["in_uniform"])
def patched(count):
    def inner(pos,item,mask,neighbours=None):
        distance=None
        if isinstance(pos,jnp.ndarray):
            if pos.shape[0]==pos.shape[1]: distance=pos
            else: pos=Vec3Array.from_array(pos)
        if distance is None: distance=(pos[:,None]-pos[None,:]).norm()
        same=item[:,None]==item[None,:]
        w=-3*jnp.log(distance+1e-6)-jnp.log(-jnp.log(U))
        return G.get_neighbours(count)(jnp.where(same,-w,jnp.inf),
                                       mask[:,None]*mask[None,:]*same, neighbours)
    return inner
G.get_random_neighbours=patched; NSB.get_random_neighbours=patched
REC=[]
_en=NSB.extract_neighbours; _ecn=NSB.extract_condition_neighbours
def ren(**k):
    f=_en(**k)
    def i(*a,**kk):
        r=f(*a,**kk); REC.append(("cur",np.asarray(r))); return r
    return i
def recn(**k):
    f=_ecn(**k)
    def i(*a,**kk):
        r=f(*a,**kk); REC.append(("cond",np.asarray(r))); return r
    return i
NSB.extract_neighbours=ren; NSB.extract_condition_neighbours=recn
cfg=configs.default_vp
P="structure_diffusion/diffusion/diffusion_stack/__layer_stack_with_state/"
def layer_params(L):
    out={}
    for k in NPZ.files:
        if k.startswith(P):
            rest=k[len(P):]; sc,nm=rest.rsplit("/",1)
            out.setdefault(sc,{})[nm]=jnp.asarray(NPZ[k][L])
    return out
pc=dict(dmap=jnp.asarray(D["in_dmap"]),dmap_mask=jnp.asarray(D["in_dmap_mask"]),
        flags=jnp.asarray(D["in_flags"]),omap=jnp.zeros((40,40,9),jnp.float32))
def fwd(f,p,pp,c,t):
    return NSB.DiffusionBlock(cfg)(f,p,pp,c,pc,t,jnp.asarray(D["in_resi"]),
        jnp.asarray(D["in_chain"]),jnp.asarray(D["in_batch"]),jnp.asarray(D["in_mask"]))
tr=hk.transform(fwd)
f=jnp.asarray(D["in_features"]); p=jnp.asarray(D["in_pos"])
pp=jnp.asarray(D["in_prev_pos"]); c=jnp.asarray(D["in_condition"]); t=jnp.asarray(D["in_time"])
traj=[]
for L in range(6):
    f,p = tr.apply(layer_params(L), jax.random.PRNGKey(1), f,p,pp,c,t)
    traj.append(np.asarray(p))
cur=np.stack([r for k,r in REC if k=="cur"]); cond=np.stack([r for k,r in REC if k=="cond"])
np.savez(sys.argv[3], out_local=np.asarray(f), out_pos=np.asarray(p),
         traj=np.stack(traj), cur=cur, cond=cond)
print("ok", cur.shape, cond.shape)
