from ..basic import get_velocity_correction


def get_corrected_wall_velocity(ux_wall, uy_wall, rho_wall=1, gx_wall=0, gy_wall=0):
    """Apply body force correction to the wall velocity.

    When body forces (e.g. gravity) are present in the simulation, the velocity
    imposed at the boundary must be shifted by half a force step to maintain
    second-order accuracy consistent with the rest of the domain.  Call this
    helper before passing the wall velocity to any boundary condition function.

    Args:
        ux_wall (scalar or jax.Array): The x-component of the prescribed wall velocity.
        uy_wall (scalar or jax.Array): The y-component of the prescribed wall velocity.
        rho_wall (scalar or jax.Array): The density at the wall (default 1).
        gx_wall (scalar or jax.Array): The x-component of the body force at the wall (default 0).
        gy_wall (scalar or jax.Array): The y-component of the body force at the wall (default 0).

    Returns:
        tuple: ``(ux_corrected, uy_corrected)`` – the velocity components after
        subtracting the half-step force contribution.
    """
    ux_corrected = ux_wall - get_velocity_correction(gx_wall, rho_wall)
    uy_corrected = uy_wall - get_velocity_correction(gy_wall, rho_wall)
    return ux_corrected, uy_corrected
