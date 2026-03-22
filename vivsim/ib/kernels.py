import jax.numpy as jnp


def kernel_peskin_3pt(r):
    """Return the 3-point discrete delta kernel.

    Args:
        r: Marker-node distance in grid units.

    Returns:
        Kernel value with the same shape as `r`.
    """
    abs_r = jnp.abs(r)
    return jnp.where(
        abs_r > 1.5,
        0,
        jnp.where(
            abs_r < 0.5,
            (1 + jnp.sqrt(1 - 3 * abs_r**2)) / 3,
            (5 - 3 * abs_r - jnp.sqrt(-2 + 6 * abs_r - 3 * abs_r**2)) / 6,
        ),
    )


def kernel_peskin_4pt(r):
    """Return the 4-point discrete delta kernel.

    Args:
        r: Marker-node distance in grid units.

    Returns:
        Kernel value with the same shape as `r`.
    """
    abs_r = jnp.abs(r)
    return jnp.where(
        abs_r > 2,
        0,
        jnp.where(
            abs_r < 1,
            (3 - 2 * abs_r + jnp.sqrt(1 + 4 * abs_r - 4 * abs_r**2)) * 0.125,
            (5 - 2 * abs_r - jnp.sqrt(-7 + 12 * abs_r - 4 * abs_r**2))* 0.125,
        ),
    )


def kernel_cosine_4pt(r):
    """Return the 4-point cosine delta kernel.

    Args:
        r: Marker-node distance in grid units.  

    Returns:
        Kernel value with the same shape as `r`.
    """

    abs_r = jnp.abs(r)
    return jnp.where(
        abs_r > 2,
        0,
        (1 + jnp.cos(jnp.pi * abs_r * 0.5)) * 0.25,
    )
