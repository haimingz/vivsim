import jax.numpy as jnp


def calculate_curl(u):
    """Compute the curl (z-component) of a 2D velocity field.

    Args:
        u (jax.Array of shape (2, NX, NY)): Velocity field.

    Returns:
        jax.Array of shape (NX, NY): The curl of the velocity field.
    """
    ux_y = jnp.gradient(u[0], axis=1)
    uy_x = jnp.gradient(u[1], axis=0)
    return ux_y - uy_x


def calculate_vorticity(u):
    """Calculate the vorticity of a 2D velocity field.

    Args:
        u (jax.Array of shape (2, NX, NY)): Velocity field.

    Returns:
        jax.Array of shape (NX, NY): The vorticity.
    """
    return calculate_curl(u)


def calculate_vorticity_dimensionless(u, l, u0):
    """Calculate the dimensionless vorticity of a velocity field.

    Args:
        u (jax.Array of shape (2, NX, NY)): Velocity field.
        l (float): The characteristic length.
        u0 (float): The free-stream velocity.

    Returns:
        jax.Array of shape (NX, NY): The dimensionless vorticity.
    """
    return calculate_curl(u) * l / u0


def calculate_velocity_magnitude(u):
    """Calculate the magnitude of a velocity field.

    Args:
        u (jax.Array of shape (2, NX, NY)): Velocity field.

    Returns:
        jax.Array of shape (NX, NY): The velocity magnitude.
    """
    return jnp.linalg.norm(u, axis=0)