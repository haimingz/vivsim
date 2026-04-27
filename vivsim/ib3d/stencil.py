import jax
import jax.numpy as jnp

from vivsim.ib.kernels import kernel_peskin_4pt


def get_ib_stencil(
    marker_coords: jax.Array,
    grid_shape: tuple[int, int, int],
    kernel=kernel_peskin_4pt,
    stencil_radius: int = 2,
) -> tuple[jax.Array, jax.Array]:
    """Return stencil weights and flattened indices for each 3D marker.

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
    stencil_indices = stencil_x_raw * (ny * nz) + stencil_y_raw * nz + stencil_z_raw

    return stencil_weights, stencil_indices
