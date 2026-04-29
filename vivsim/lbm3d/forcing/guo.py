"""Guo forcing scheme for the D3Q19 lattice."""

import jax.numpy as jnp

from ..lattice import D3Q19
from ..collision.mrt import M, M_INV, get_mrt_relaxation_matrix


D3Q19_INV_CS2 = 3.0
D3Q19_INV_CS4 = 9.0


def get_guo_forcing_term(g, u):
    """Compute the D3Q19 lattice forcing term using Guo's scheme."""

    ux, uy, uz = u
    gx, gy, gz = g

    velocity_projection = jnp.stack([
        jnp.zeros_like(ux),
        ux, -ux, uy, -uy, uz, -uz,
        ux + uy, -ux + uy, ux - uy, -ux - uy,
        ux + uz, -ux + uz, ux - uz, -ux - uz,
        uy + uz, -uy + uz, uy - uz, -uy - uz,
    ])
    force_projection = jnp.stack([
        jnp.zeros_like(gx),
        gx, -gx, gy, -gy, gz, -gz,
        gx + gy, -gx + gy, gx - gy, -gx - gy,
        gx + gz, -gx + gz, gx - gz, -gx - gz,
        gy + gz, -gy + gz, gy - gz, -gy - gz,
    ])
    force_velocity = jnp.sum(g * u, axis=0)

    return D3Q19.w[:, None, None, None] * (
        D3Q19_INV_CS2 * (force_projection - force_velocity[None, ...])
        + D3Q19_INV_CS4 * force_projection * velocity_projection
    )


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
