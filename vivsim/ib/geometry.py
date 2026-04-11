"""Geometry helpers for immersed-boundary methods (2D and 3D)."""

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# 2D curve geometry
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 3D surface-mesh geometry
# ---------------------------------------------------------------------------


def get_triangle_areas(vertex_coords, faces):
    """Return triangle areas for a triangulated surface mesh.

    Args:
        vertex_coords: Vertex coordinates with shape ``(n_vertices, 3)``.
        faces: Triangle connectivity with shape ``(n_faces, 3)``.

    Returns:
        jax.Array: Triangle areas with shape ``(n_faces,)``.
    """
    triangles = jnp.asarray(vertex_coords)[jnp.asarray(faces)]
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    return 0.5 * jnp.linalg.norm(jnp.cross(edge_1, edge_2), axis=1)


def get_surface_area(vertex_coords, faces):
    """Return total area of a triangulated surface mesh."""
    return jnp.sum(get_triangle_areas(vertex_coords, faces))


def get_vertex_dA(vertex_coords, faces):
    """Return lumped surface-area weights for mesh vertices.

    Each triangle area is distributed equally to its three vertices.

    Args:
        vertex_coords: Vertex coordinates with shape ``(n_vertices, 3)``.
        faces: Triangle connectivity with shape ``(n_faces, 3)``.

    Returns:
        jax.Array: Vertex area weights with shape ``(n_vertices,)``.
    """
    vertex_coords = jnp.asarray(vertex_coords)
    faces = jnp.asarray(faces)
    triangle_areas = get_triangle_areas(vertex_coords, faces)
    vertex_dA = jnp.zeros((vertex_coords.shape[0],), dtype=vertex_coords.dtype)
    return vertex_dA.at[faces.reshape(-1)].add(
        jnp.repeat(triangle_areas / 3.0, 3)
    )


def get_marker_da(vertex_coords, faces):
    """Compatibility alias for vertex surface-area weights."""
    return get_vertex_dA(vertex_coords, faces)
