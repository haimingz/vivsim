import jax 
import jax.numpy as jnp


@jax.jit
def calculate_curl(u):
    ux_y = jnp.gradient(u[0], axis=1)
    uy_x = jnp.gradient(u[1], axis=0)
    curl = ux_y - uy_x
    return curl