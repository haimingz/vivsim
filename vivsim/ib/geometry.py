"""Geometry helpers for immersed-boundary methods (2D)."""

import jax.numpy as jnp


def get_area(marker_coords):
    """Return the area of a closed polygon.

    Args:
        marker_coords: Polygon vertices, shape `(n_markers, 2)`.

    Returns:
        Polygon area as a scalar array.
    """
    x = marker_coords[:, 0]
    y = marker_coords[:, 1]
    polygon_area = 0.5 * jnp.abs(jnp.sum(x * jnp.roll(y, 1) - y * jnp.roll(x, 1)))
    return polygon_area


def get_ds_closed(marker_coords):
    """Return segment lengths for markers on a closed curve.

    Args:
        marker_coords: Marker coordinates, shape `(n_markers, 2)`.

    Returns:
        Marker segment lengths, shape `(n_markers,)`.
    """
    segment_lengths = jnp.linalg.norm(
        marker_coords - jnp.roll(marker_coords, shift=-1, axis=0), axis=1
    )
    marker_ds = (segment_lengths + jnp.roll(segment_lengths, shift=1)) / 2.0
    return marker_ds


def get_ds_open(marker_coords):
    """Return segment lengths for markers on an open curve.

    Args:
        marker_coords: Marker coordinates, shape `(n_markers, 2)`.

    Returns:
        Marker segment lengths, shape `(n_markers,)`.
    """
    segment_lengths = jnp.linalg.norm(
        marker_coords[1:] - marker_coords[:-1], axis=1
    )
    marker_ds = jnp.pad(segment_lengths / 2, (1, 0)) + jnp.pad(
        segment_lengths / 2, (0, 1)
    )
    return marker_ds
