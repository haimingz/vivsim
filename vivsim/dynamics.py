import jax.numpy as jnp



def newmark(a, v, d, g, m, k, c, alpha=0.5, beta=0.25):
    """Newmark-beta method for dynamics"""
    v1 = v + (1 - alpha) * a
    v2 = d + v + (0.5 - beta) * a
    a_next = (g - c * v1 - k * v2) / (m + alpha * c + beta * k)
    v_next = alpha * a_next + v1
    d_next = beta * a_next + v2
    return a_next, v_next, d_next
