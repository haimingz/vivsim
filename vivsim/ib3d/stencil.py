"""Interpolation and spreading stencils for the 3D immersed-boundary method."""

import jax
import jax.numpy as jnp

from .kernels import kernel_peskin_4pt


def get_ib_stencil(
    marker_coords: jax.Array,
    grid_shape: tuple[int, int, int],
    kernel=kernel_peskin_4pt,
    stencil_radius: int = 2,
) -> tuple[jax.Array, jax.Array]:
    """Return 3D stencil weights and flattened indices for each marker.

    Args:
        marker_coords: Marker coordinates with shape ``(n_markers, 3)``.
        grid_shape: Eulerian grid shape ``(nx, ny, nz)``.
        kernel: One-dimensional discrete delta kernel.
        stencil_radius: Half-width of the tensor-product stencil.

    Returns:
        tuple[jax.Array, jax.Array]:
            ``(stencil_weights, stencil_indices)`` with shapes
            ``(n_markers, n_stencil)`` and ``(n_markers, n_stencil)``.
    """

    if len(grid_shape) != 3:
        raise ValueError(f"grid_shape must be a 3-tuple, got {grid_shape}.")

    marker_coords = jnp.asarray(marker_coords)
    if marker_coords.ndim != 2 or marker_coords.shape[1] != 3:
        raise ValueError(
            "marker_coords must have shape (n_markers, 3), "
            f"got {marker_coords.shape}."
        )

    nx, ny, nz = grid_shape
    stencil_offsets = jnp.arange(
        -stencil_radius + 1, stencil_radius + 1, dtype=jnp.int32
    )
    stencil_offset_x, stencil_offset_y, stencil_offset_z = jnp.meshgrid(
        stencil_offsets, stencil_offsets, stencil_offsets, indexing="ij"
    )
    stencil_offset_x = stencil_offset_x.reshape(1, -1)
    stencil_offset_y = stencil_offset_y.reshape(1, -1)
    stencil_offset_z = stencil_offset_z.reshape(1, -1)

    stencil_ref = jnp.floor(marker_coords).astype(jnp.int32)
    stencil_x_raw = stencil_ref[:, 0:1] + stencil_offset_x
    stencil_y_raw = stencil_ref[:, 1:2] + stencil_offset_y
    stencil_z_raw = stencil_ref[:, 2:3] + stencil_offset_z

    stencil_weights = (
        kernel(stencil_x_raw - marker_coords[:, 0:1])
        * kernel(stencil_y_raw - marker_coords[:, 1:2])
        * kernel(stencil_z_raw - marker_coords[:, 2:3])
    )
    # Wrap only the Eulerian target nodes. The kernel distances must keep the
    # unwrapped local coordinates so markers near periodic faces still see the
    # intended compact support.
    stencil_x = stencil_x_raw % nx
    stencil_y = stencil_y_raw % ny
    stencil_z = stencil_z_raw % nz
    stencil_indices = stencil_x * (ny * nz) + stencil_y * nz + stencil_z

    return stencil_weights, stencil_indices


def interpolate(
    grid_values: jax.Array,
    stencil_weights: jax.Array,
    stencil_indices: jax.Array,
) -> jax.Array:
    """Interpolate a 3D Eulerian field to marker locations.

    Args:
        grid_values: Grid field with shape ``(n_components, nx, ny, nz)``.
        stencil_weights: Stencil weights with shape ``(n_markers, n_stencil)``.
        stencil_indices: Flattened stencil indices with the same shape as
            ``stencil_weights``.

    Returns:
        jax.Array: Marker values with shape ``(n_markers, n_components)``.
    """

    flat_grid_values = grid_values.reshape(grid_values.shape[0], -1)
    stencil_values = flat_grid_values[:, stencil_indices]
    return jnp.einsum("ms,cms->mc", stencil_weights, stencil_values, precision='highest')


def spread(
    marker_values: jax.Array,
    grid_values: jax.Array,
    stencil_weights: jax.Array,
    stencil_indices: jax.Array,
) -> jax.Array:
    """Spread marker values onto a 3D Eulerian field.

    Args:
        marker_values: Marker values with shape ``(n_markers, n_components)``.
        grid_values: Grid field to accumulate into, shape
            ``(n_components, nx, ny, nz)``.
        stencil_weights: Stencil weights with shape ``(n_markers, n_stencil)``.
        stencil_indices: Flattened stencil indices with the same shape as
            ``stencil_weights``.

    Returns:
        jax.Array: Updated grid field with the same shape as ``grid_values``.
    """

    grid_shape = grid_values.shape[1:]
    flat_grid_values = grid_values.reshape(grid_values.shape[0], -1)
    stencil_values = jnp.einsum("mc,ms->cms", marker_values, stencil_weights, precision='highest')
    flat_grid_values = flat_grid_values.at[:, stencil_indices].add(stencil_values)
    return flat_grid_values.reshape(flat_grid_values.shape[0], *grid_shape)
