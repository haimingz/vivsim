"""Multi-direct forcing for 3D immersed-boundary coupling."""

import jax
import jax.numpy as jnp
from jax import lax

from ..lbm3d import get_velocity_correction
from .stencil import interpolate, spread


def multi_direct_forcing(
    grid_u: jax.Array,
    stencil_weights: jax.Array,
    stencil_indices: jax.Array,
    marker_u_target: jax.Array,
    marker_dA: jax.Array,
    n_iter: int = 5,
) -> tuple[jax.Array, jax.Array]:
    """Run 3D multi-direct forcing and return fluid and marker forces.

    Args:
        grid_u: Fluid velocity at grid points, shape ``(3, nx, ny, nz)``.
        stencil_weights: Stencil weights, shape ``(n_markers, n_stencil)``.
        stencil_indices: Flattened stencil indices, same shape as
            ``stencil_weights``.
        marker_u_target: Target fluid velocity at marker positions, shape
            ``(n_markers, 3)``.
        marker_dA: Marker surface-area weights, scalar or shape
            ``(n_markers,)``.
        n_iter: Number of MDF correction iterations.

    Returns:
        tuple[jax.Array, jax.Array]:
            ``(grid_force, marker_reaction_force)`` with shapes
            ``(3, nx, ny, nz)`` and ``(n_markers, 3)``.
    """

    marker_dA = jnp.asarray(marker_dA).reshape(-1, 1)
    grid_force_base = jnp.zeros_like(grid_u)
    marker_force_total = jnp.zeros_like(marker_u_target)

    marker_u = interpolate(grid_u, stencil_weights, stencil_indices)

    def body_fun(_: int, state: tuple[jax.Array, jax.Array]):
        marker_u, marker_force_total = state

        marker_force_step = (marker_u_target - marker_u) * (marker_dA * 2.0)
        marker_force_total += marker_force_step

        grid_force_step = spread(
            marker_force_step, grid_force_base, stencil_weights, stencil_indices
        )
        grid_u_step = get_velocity_correction(grid_force_step)
        marker_u += interpolate(grid_u_step, stencil_weights, stencil_indices)

        return marker_u, marker_force_total

    marker_u, marker_force_total = lax.fori_loop(
        0, n_iter, body_fun, (marker_u, marker_force_total)
    )

    marker_reaction_force = -marker_force_total
    grid_force = spread(
        marker_force_total, grid_force_base, stencil_weights, stencil_indices
    )

    return grid_force, marker_reaction_force

