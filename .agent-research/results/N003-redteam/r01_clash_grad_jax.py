import numpy as np, jax, jax.numpy as jnp
ca = np.load("/tmp/r01_ca.npy")
def cost(x):
    dist = jnp.sqrt((jnp.maximum(x[:, None] - x[None, :], 1e-3) ** 2).sum(axis=-1))
    switch = jnp.maximum(1 - ((dist - 2) / 8) ** 6, 1e-6) / jnp.maximum(1 - ((dist - 2) / 8) ** 12, 1e-6)
    return switch.mean()
g = np.asarray(jax.grad(cost)(jnp.array(ca)))
g = np.where(np.isnan(g), 0, g)
np.save("/tmp/r01_jax.npy", g)
t = np.load("/tmp/r01_torch.npy")
print("jax   grad norm", np.linalg.norm(g), "max", np.abs(g).max())
print("torch grad norm", np.linalg.norm(t), "max", np.abs(t).max())
print("max|d| =", np.abs(g - t).max())
den = max(np.abs(g).max(), 1e-12)
print("rel    =", np.abs(g - t).max() / den)
c = np.corrcoef(g.ravel(), t.ravel())[0, 1]
print("corr   =", c)
