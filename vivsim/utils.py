import jax.numpy as jnp


def smooth_step(t, t_max):
    """
    A smooth step function that ramps from 0 to 1 over `t_max` steps
    using a half-cosine wave. Useful for avoiding impulsive start waves
    in fluid simulations.

    Args:
        t (int or float): Current time step.
        t_max (int or float): The time step when it reaches 1.0.

    Returns:
        float: A value between 0.0 and 1.0.
    """
    return jnp.where(t < t_max, 0.5 * (1.0 - jnp.cos(jnp.pi * t / t_max)), 1.0)
