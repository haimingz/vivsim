"""
This module implements the Non-Equilibrium Extrapolation (NEE) boundary condition
for the Lattice Boltzmann Method (LBM) in a 2D D2Q9 lattice.

The body force parameters gx_wall and gy_wall represent the external body forces applied
at the boundary (e.g., gravity or other acceleration fields). These are used to correct
the wall velocity when body forces are present in the simulation.
"""



from ..basic import LEFT_DIRS, RIGHT_DIRS, UP_DIRS, DOWN_DIRS, get_velocity_correction
import jax.numpy as jnp
from .. import get_equilibrium, get_macroscopic


def boundary_nee(f, loc: str, rho_wall=1, ux_wall=0, uy_wall=0, gx_wall=0, gy_wall=0):
    """Non-Equilibrium Extrapolation scheme.

    This is the core function for the nee_velocity and nee_pressure functions.

    It applies the Non-Equilibrium Extrapolation scheme to the distribution function
    at the specified boundary location, enforcing the given wall velocity and density.
    It modifies all the distribution functions at the boundary in such a way that
    ensures a zero-gradient condition for the non-equilibrium part of the
    distribution functions at the boundary.

    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        loc (str): The boundary where the velocity condition is enforced,
            can be 'left', 'right', 'top', or 'bottom'.
        rho_wall (scalar or jax.Array of shape NX or NY): The density at the wall.
        ux_wall (scalar or jax.Array of shape NX or NY): The x-component of velocity.
        uy_wall (scalar or jax.Array of shape NX or NY): The y-component of velocity.
        gx_wall (scalar or jax.Array of shape NX or NY): The x-component of body force.
        gy_wall (scalar or jax.Array of shape NX or NY): The y-component of body force.

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """

    ux_wall = ux_wall - get_velocity_correction(gx_wall, rho_wall)
    uy_wall = uy_wall - get_velocity_correction(gy_wall, rho_wall)
    
    u_wall = jnp.array([ux_wall, uy_wall])
    feq_wall = get_equilibrium(rho_wall, u_wall)

    if loc == "left":
        rho_next, u_next = get_macroscopic(f[:, 1])
        fneq_next = f[:, 1] - get_equilibrium(rho_next, u_next)
        f = f.at[:, 0].set(feq_wall + fneq_next)

    elif loc == "right":
        rho_next, u_next = get_macroscopic(f[:, -3:-1])
        fneq_next = f[:, -3:-1] - get_equilibrium(rho_next, u_next)
        # f = f.at[:, -1].set(feq_wall + fneq_next[:,-1])
        fneq_wall = jnp.mean(fneq_next, axis=1)
        f = f.at[:, -1].set(feq_wall + fneq_wall)

    elif loc == "top":
        rho_next, u_next = get_macroscopic(f[:, :, -2])
        fneq_next = f[:, :, -2] - get_equilibrium(rho_next, u_next)
        f = f.at[:, :, -1].set(feq_wall + fneq_next)

    elif loc == "bottom":
        rho_next, u_next = get_macroscopic(f[:, :, 1])
        fneq_next = f[:, :, 1] - get_equilibrium(rho_next, u_next)
        f = f.at[:, :, 0].set(feq_wall + fneq_next)

    return f


def boundary_velocity_nee(f, loc: str, ux_wall=0, uy_wall=0, gx_wall=0, gy_wall=0):
    """Enforce given velocity ux_wall, uy_wall at the specified boundary using the
    Non-Equilibrium Extrapolation scheme.

    This function should be called after the streaming step. In this function,
    the density at the wall is computed based on the known outgoing distribution functions.

    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        loc (str): The boundary where the velocity condition is enforced,
            can be 'left', 'right', 'top', or 'bottom'.
        ux_wall (scalar or jax.Array of shape NX or NY): The x-component of velocity.
        uy_wall (scalar or jax.Array of shape NX or NY): The y-component of velocity.
        gx_wall (scalar or jax.Array of shape NX or NY): The x-component of body force.
        gy_wall (scalar or jax.Array of shape NX or NY): The y-component of body force.

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """

    if loc not in ["left", "right", "top", "bottom"]:
        raise ValueError(
            "Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'."
        )

    if loc == "left":
        rho_wall = (f[0, 0] + f[2, 0] + f[4, 0] + 2 * (f[3, 0] + f[6, 0] + f[7, 0])) / (1 - ux_wall)

    elif loc == "right":
        rho_wall = (f[0, -1] + f[2, -1] + f[4, -1] + 2 * (f[1, -1] + f[5, -1] + f[8, -1])) / (1 + ux_wall)

    elif loc == "top":
        rho_wall = (f[0, :, -1] + f[1, :, -1] + f[3, :, -1] + 2 * (f[2, :, -1] + f[5, :, -1] + f[6, :, -1])) / (1 + uy_wall)

    elif loc == "bottom":
        rho_wall = (f[0, :, 0] + f[1, :, 0] + f[3, :, 0] + 2 * (f[4, :, 0] + f[7, :, 0] + f[8, :, 0])) / (1 - uy_wall)

    return boundary_nee(f, loc, rho_wall=rho_wall, ux_wall=ux_wall, uy_wall=uy_wall, gx_wall=gx_wall, gy_wall=gy_wall)


def boundary_pressure_nee(f, loc: str, rho_wall=1, gx_wall=0, gy_wall=0):
    """Enforce given pressure (density) rho_wall at the specified boundary using the
    Non-Equilibrium Extrapolation scheme.

    This function should be called after the streaming step. In this function,
    the velocity normal to the boundary is computed based on the known outgoing distribution functions.
    The velocity tangential to the boundary are taken from the fluid nodes next to the boundary.

    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        loc (str): The boundary where the pressure condition is enforced,
            can be 'left', 'right', 'top', or 'bottom'.
        rho_wall (scalar or jax.Array of shape NX or NY): The density at the wall.
        gx_wall (scalar or jax.Array of shape NX or NY): The x-component of body force.
        gy_wall (scalar or jax.Array of shape NX or NY): The y-component of body force.

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """

    if loc not in ["left", "right", "top", "bottom"]:
        raise ValueError(
            "Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'."
        )

    if loc == "left":
        ux_wall = (f[0, 0] + f[2, 0] + f[4, 0] + 2 * (f[3, 0] + f[6, 0] + f[7, 0])) / rho_wall - 1
        rho_next = jnp.sum(f[:, 1], axis=0)
        uy_wall = (jnp.sum(f[UP_DIRS, 1], axis=0) - jnp.sum(f[DOWN_DIRS, 1], axis=0)) / rho_next

    elif loc == "right":
        ux_wall = (f[0, -1] + f[2, -1] + f[4, -1] + 2 * (f[1, -1] + f[5, -1] + f[8, -1])) / rho_wall - 1
        rho_next = jnp.sum(f[:, -2], axis=0)
        uy_wall = (jnp.sum(f[UP_DIRS, -2], axis=0) - jnp.sum(f[DOWN_DIRS, -2], axis=0)) / rho_next

    elif loc == "top":
        uy_wall = (f[0, :, -1] + f[1, :, -1] + f[3, :, -1] + 2 * (f[2, :, -1] + f[5, :, -1] + f[6, :, -1])) / rho_wall - 1
        rho_next = jnp.sum(f[:, :, -2], axis=0)
        ux_wall = (jnp.sum(f[RIGHT_DIRS, :, -2], axis=0) - jnp.sum(f[LEFT_DIRS, :, -2], axis=0)) / rho_next

    elif loc == "bottom":
        uy_wall = (f[0, :, 0] + f[1, :, 0] + f[3, :, 0] + 2 * (f[4, :, 0] + f[7, :, 0] + f[8, :, 0])) / rho_wall - 1
        rho_next = jnp.sum(f[:, :, 1], axis=0)
        ux_wall = (jnp.sum(f[RIGHT_DIRS, :, 1], axis=0) - jnp.sum(f[LEFT_DIRS, :, 1], axis=0)) / rho_next

    return boundary_nee(f, loc, rho_wall=rho_wall, ux_wall=ux_wall, uy_wall=uy_wall, gx_wall=gx_wall, gy_wall=gy_wall)

