"""Exact Difference Method (EDM) forcing scheme for the D3Q19 lattice."""

import jax.numpy as jnp

from .guo import get_guo_forcing_term


def forcing_edm(f, g, u, rho):
    """Apply the D3Q19 Exact Difference Method forcing scheme.

    The EDM applies external forcing through the difference between equilibrium
    distributions evaluated at force-shifted velocities. In this implementation,
    the result is written in the algebraically equivalent form

    ``rho * get_guo_forcing_term(g, u)``.

    This makes the forcing step independent of the collision model, which is
    convenient when using MRT, regularized, or KBC-style collisions.

    Args:
        f (jax.Array): Distribution function with shape ``(19, *spatial_dims)``.
        g (jax.Array): External force density with shape ``(3, *spatial_dims)``.
        u (jax.Array): Fluid velocity with shape ``(3, *spatial_dims)``.
        rho (jax.Array): Fluid density with shape ``(*spatial_dims)``.

    Returns:
        jax.Array: Updated distribution function with the same shape as ``f``.
    """

    return f + rho[None, ...] * get_guo_forcing_term(g, u)
