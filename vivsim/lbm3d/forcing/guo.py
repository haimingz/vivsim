"""Guo forcing scheme for the D3Q19 lattice."""

import jax.numpy as jnp

from ..lattice import D3Q19
from ..collision.mrt import M, M_INV, get_mrt_relaxation_matrix


def get_guo_forcing_term(g, u):
    """Compute the D3Q19 lattice forcing term using Guo's scheme."""

    uc = (
        u[0] * D3Q19.c[:, 0, None, None, None]
        + u[1] * D3Q19.c[:, 1, None, None, None]
        + u[2] * D3Q19.c[:, 2, None, None, None]
    )

    g_lattice = D3Q19.w[:, None, None, None] * (
        g[0] * (
            (D3Q19.c[:, 0, None, None, None] - u[None, 0]) / D3Q19.cs2
            + D3Q19.c[:, 0, None, None, None] * uc / (D3Q19.cs2**2)
        )
        + g[1] * (
            (D3Q19.c[:, 1, None, None, None] - u[None, 1]) / D3Q19.cs2
            + D3Q19.c[:, 1, None, None, None] * uc / (D3Q19.cs2**2)
        )
        + g[2] * (
            (D3Q19.c[:, 2, None, None, None] - u[None, 2]) / D3Q19.cs2
            + D3Q19.c[:, 2, None, None, None] * uc / (D3Q19.cs2**2)
        )
    )
    return g_lattice


def forcing_guo_bgk(f, g, u, omega):
    """Apply Guo forcing to a BGK / SRT collision step.

    Args:
        f (jax.Array): Distribution function with shape ``(19, *spatial_dims)``.
        g (jax.Array): External force density with shape ``(3, *spatial_dims)``.
        u (jax.Array): Fluid velocity with shape ``(3, *spatial_dims)``.
        omega (float): BGK relaxation parameter.

    Returns:
        jax.Array: Updated distribution function with applied forcing.
    """

    g_lattice = get_guo_forcing_term(g, u)
    return f + g_lattice * (1 - 0.5 * omega)


def get_mrt_forcing_operator(omega):
    """Pre-compute the MRT forcing operator for D3Q19.

    The imported MRT basis is validated against the expected ``(19, 19)``
    D3Q19 shape before constructing the forcing operator.
    """

    S = get_mrt_relaxation_matrix(omega)

    return M_INV @ (jnp.eye(D3Q19.q) - 0.5 * S) @ M


def forcing_guo_mrt(f, g, u, mrt_forcing_operator):
    """Apply Guo forcing to an MRT collision step.

    Args:
        f (jax.Array): Distribution function with shape ``(19, *spatial_dims)``.
        g (jax.Array): External force density with shape ``(3, *spatial_dims)``.
        u (jax.Array): Fluid velocity with shape ``(3, *spatial_dims)``.
        mrt_forcing_operator (jax.Array): MRT forcing operator with shape
            ``(19, 19)``.

    Returns:
        jax.Array: Updated distribution function with applied forcing.
    """

    g_lattice = get_guo_forcing_term(g, u)
    return f + jnp.tensordot(
        mrt_forcing_operator, g_lattice, axes=([1], [0]), precision='highest'
    )
