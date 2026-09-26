import jax.numpy as jnp
from salad.inference.symmetry import Screw
x=jnp.arange(4*5*3,dtype=jnp.float32).reshape(4,5,3)
for count in (1,4,5):
    s=Screw(count=count,angle=360/count,radius=10,translation=0)
    try:
        y=s.replicate_pos(s.select_pos(x,True),True)
        print('JAX',count,'input',len(x),'output',len(y),'identity',bool(jnp.array_equal(x,y)))
    except Exception as e: print('JAX',count,type(e).__name__,str(e)[:200])
