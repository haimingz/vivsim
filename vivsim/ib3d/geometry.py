"""Geometry helpers for immersed-boundary methods (3D)."""

import jax.numpy as jnp


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


def get_volume(vertex_coords, faces):
    """Return the volume enclosed by a closed triangulated surface mesh.

    The mesh faces should have consistent winding. The absolute value is
    returned so outward and inward orientation give the same volume.

    Args:
        vertex_coords: Vertex coordinates with shape ``(n_vertices, 3)``.
        faces: Triangle connectivity with shape ``(n_faces, 3)``.

    Returns:
        jax.Array: Enclosed volume as a scalar array.
    """
    triangles = jnp.asarray(vertex_coords)[jnp.asarray(faces)]
    signed_volumes = jnp.einsum(
        "ij,ij->i",
        triangles[:, 0],
        jnp.cross(triangles[:, 1], triangles[:, 2]),
    ) / 6.0
    return jnp.abs(jnp.sum(signed_volumes))


def get_ds(vertex_coords, faces):
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
