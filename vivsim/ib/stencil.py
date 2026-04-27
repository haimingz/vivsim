import jax
import jax.numpy as jnp

from .kernels import kernel_peskin_4pt


def get_ib_stencil(
    marker_x: jax.Array,
    marker_y: jax.Array,
    ny: int,
    kernel=kernel_peskin_4pt,
    stencil_radius: int = 2,
) -> tuple[jax.Array, jax.Array]:
    """Return stencil weights and flattened indices for each marker.

    Args:
        marker_x: Marker x-coordinates, shape `(n_markers,)`.
        marker_y: Marker y-coordinates, shape `(n_markers,)`.
        ny: Grid size along the y-direction.
        kernel: One-dimensional delta kernel.
        stencil_radius: Half-width of the tensor-product stencil.

    Returns:
        Tuple `(stencil_weights, stencil_indices)`, both with shape
        `(n_markers, n_stencil)`.
    """
    grid_ny = ny
    stencil_offsets = jnp.arange(
        -stencil_radius + 1, stencil_radius + 1, dtype=jnp.int32
    )   
    stencil_offset_x, stencil_offset_y = jnp.meshgrid(
        stencil_offsets, stencil_offsets, indexing="ij"
    )
    stencil_offset_x = stencil_offset_x.reshape(1, -1)
    stencil_offset_y = stencil_offset_y.reshape(1, -1)

    marker_x = jnp.asarray(marker_x)
    marker_y = jnp.asarray(marker_y)

    stencil_ref_x = jnp.floor(marker_x).astype(jnp.int32)
    stencil_ref_y = jnp.floor(marker_y).astype(jnp.int32)

    stencil_x = stencil_ref_x[:, None] + stencil_offset_x
    stencil_y = stencil_ref_y[:, None] + stencil_offset_y

    stencil_weights = kernel(stencil_x - marker_x[:, None]) * kernel(
        stencil_y - marker_y[:, None]
    )
    stencil_indices = stencil_x * grid_ny + stencil_y

    return stencil_weights, stencil_indices


def interpolate(
    grid_values: jax.Array,
    stencil_weights: jax.Array,
    stencil_indices: jax.Array,
) -> jax.Array:
    """Interpolate a grid field to marker locations.

    Args:
        grid_values: Grid field, shape `(n_components, nx, ny)`.
        stencil_weights: Stencil weights, shape `(n_markers, n_stencil)`.
        stencil_indices: Flattened stencil indices, same shape as
            `stencil_weights`.

    Note: 
        The `grid_values` array is flattened to shape `(n_components, nx*ny)` 
        for efficient indexing. The `stencil_weights` and `stencil_indices` 
        are precomputed using `get_ib_stencil`.


    Returns:
        Marker values, shape `(n_markers, n_components)`.
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
    """Spread marker values onto a grid field.

    Args:
        marker_values: Marker values, shape `(n_markers, n_components)`.
        grid_values: Grid field to accumulate into, shape `(n_components, nx, ny)`.
        stencil_weights: Stencil weights, shape `(n_markers, n_stencil)`.
        stencil_indices: Flattened stencil indices, same shape as
            `stencil_weights`.
    
    Note: 
        The `grid_values` array is flattened to shape `(n_components, nx*ny)` 
        for efficient indexing. The `stencil_weights` and `stencil_indices` 
        are precomputed using `get_ib_stencil`.

    Returns:
        Updated grid field with the same shape as `grid_values`.
    """
    grid_shape = grid_values.shape[1:]
    flat_grid_values = grid_values.reshape(grid_values.shape[0], -1)
    stencil_values = jnp.einsum("mc,ms->cms", marker_values, stencil_weights, precision='highest')
    flat_grid_values = flat_grid_values.at[:, stencil_indices].add(
        stencil_values
    )
    return flat_grid_values.reshape(flat_grid_values.shape[0], *grid_shape)



