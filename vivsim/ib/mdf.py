"""Multi-direct forcing for immersed-boundary coupling."""

import jax
import jax.numpy as jnp
from jax import lax

from .stencil import interpolate, spread


def multi_direct_forcing(
    grid_u: jax.Array,
    stencil_weights: jax.Array,
    stencil_indices: jax.Array,
    marker_u_target: jax.Array,
    marker_ds: jax.Array,
    n_iter: int = 5,
) -> tuple[jax.Array, jax.Array]:
    """Run multi-direct forcing and return fluid and marker forces.

    Works for both 2D and 3D grids. The dimensionality is inferred from
    ``grid_u`` (e.g. shape ``(2, nx, ny)`` or ``(3, nx, ny, nz)``).

    Args:
        grid_u: Fluid velocity at grid points.
        stencil_weights: Stencil weights, shape ``(n_markers, n_stencil)``.
        stencil_indices: Flattened stencil indices, same shape as
            ``stencil_weights``.
        marker_u_target: Target fluid velocity at marker positions.
        marker_ds: Marker spacing weights, scalar or shape ``(n_markers,)``.
        n_iter: Number of MDF correction iterations.

    Returns:
        Tuple ``(grid_force, marker_reaction_force)``.
    """
    marker_ds = jnp.asarray(marker_ds).reshape(-1, 1)
    grid_force_base = jnp.zeros_like(grid_u)
    marker_force_total = jnp.zeros_like(marker_u_target)

    marker_u = interpolate(grid_u, stencil_weights, stencil_indices)

    def body_fun(_: int, state: tuple[jax.Array, jax.Array]):
        marker_u, marker_force_total = state

        marker_force_step = (marker_u_target - marker_u) * (marker_ds * 2.0)
        marker_force_total += marker_force_step

        grid_force_step = spread(
            marker_force_step, grid_force_base, stencil_weights, stencil_indices
        )
        grid_u_step = grid_force_step * 0.5  # inline velocity correction
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
