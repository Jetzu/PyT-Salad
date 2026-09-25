"""JAX-side of R02: verbatim upstream screw_replicate."""
import numpy as np, jax, jax.numpy as jnp
def cyclic_centering(x, docenter=True):
    if not docenter: return x
    return x.at[:].add(-x[:, 1].mean(axis=0))
def screw_replicate(x, count=3, angle=0, radius=15.0, translation=0.0,
                    chain_center=False, time=0.0, mean=True):
    rotmat = jnp.array([[np.cos(angle),0.,-np.sin(angle)],[0.,1.,0.],[np.sin(angle),0.,np.cos(angle)]])
    rotmat_x = [jnp.eye(3), rotmat]
    for idx in range(count - 2):
        rotmat_x.append(jnp.einsum("ac,cb->ab", rotmat_x[-1], rotmat))
    size = x.shape[0] // count
    if mean:
        first = 0
        for idx in range(count):
            first += jnp.einsum("...c,cd->...d", cyclic_centering(x[idx*size:(idx+1)*size]), rotmat_x[idx].T)
        first /= count
    else:
        idx = count // 2
        first = jnp.einsum("...c,cd->...d", cyclic_centering(x[idx*size:(idx+1)*size]), rotmat_x[idx].T)
    def cost(x):
        dist = jnp.sqrt((jnp.maximum(x[:,None]-x[None,:],1e-3)**2).sum(axis=-1))
        s = jnp.maximum(1-((dist-2)/8)**6,1e-6)/jnp.maximum(1-((dist-2)/8)**12,1e-6)
        return s.mean()
    grad = jax.grad(cost, argnums=0)(first[:, 1])
    grad = jnp.where(jnp.isnan(grad), 0, grad)
    if chain_center: first += time * 5 * grad[:, None]
    first = first.at[:].add(jnp.array([radius, 0.0, 0.0]))
    reps = [jnp.einsum("...c,cd->...d", first, rotmat_x[i]) + i*jnp.array([0.,translation,0.]) for i in range(count)]
    result = jnp.concatenate(reps, axis=0)
    return result - result[:, 1].mean(axis=0)

D = np.load("/tmp/r02.npz")
names = sorted({k.split("|")[0] for k in D.files})
for n in names:
    x, yt, p = D[n+"|x"], D[n+"|y"], D[n+"|p"]
    L, cnt, ang, rad, tr, cc, mn, tm = p
    yj = np.asarray(screw_replicate(jnp.array(x), count=int(cnt), angle=ang/180*np.pi,
                                    radius=rad, translation=tr, chain_center=bool(cc),
                                    time=tm, mean=bool(mn)))
    if yj.shape != yt.shape:
        print(f"{n:<22} SHAPE  jax{yj.shape} torch{yt.shape}")
        k = min(yj.shape[0], yt.shape[0])
        d = np.abs(yj[:k]-yt[:k]).max()
        print(f"{'':<22}   overlap[0:{k}] max|d| = {d:.4g}")
        continue
    d = np.abs(yj-yt).max()
    print(f"{n:<22} {'OK ' if d<1e-3 else 'DIFF'} max|d| = {d:.6g}   (scale {np.abs(yj).max():.3g})")
