import jax.numpy as jnp

def newmark(a, v, d, g, m, k, c, dt=1, gamma=0.5, beta=0.25):
    """Newmark-beta method for dynamics"""

    c1, c2 = gamma * dt, beta * dt**2
    v1 = v + dt * (1 - gamma) * a
    v2 = d + dt * v + dt**2 * (0.5 - beta) * a
    a_next = (g - c * v1 - k * v2) / (m + c1 * c + c2 * k)
    v_next = c1 * a_next + v1
    d_next = c2 * a_next + v2
    return a_next, v_next, d_next
