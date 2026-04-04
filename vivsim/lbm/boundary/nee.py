"""
This module implements the Non-Equilibrium Extrapolation (NEE) boundary condition
for the Lattice Boltzmann Method (LBM) in a 2D D2Q9 lattice.

When body forces are present, pass the corrected wall velocity obtained from
``get_corrected_wall_velocity`` (from this package) before calling these functions.
"""

from ..basic import LEFT_DIRS, RIGHT_DIRS, UP_DIRS, DOWN_DIRS
import jax.numpy as jnp
from .. import get_equilibrium, get_macroscopic
from ._slices import (
    WALL_SLICE, NEIGHBOR_SLICE,
    SPATIAL_NEIGHBOR_SLICE,
    BOUNDARY_SIZE_AXIS, DENSITY_RECOVERY, validate_loc,
)


def boundary_nee(f, loc: str, rho_wall=1, ux_wall=0, uy_wall=0):
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

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """
    validate_loc(loc)

    size = f.shape[BOUNDARY_SIZE_AXIS[loc]]

    if jnp.isscalar(rho_wall):
        rho_wall = jnp.full(size, rho_wall)
    if jnp.isscalar(ux_wall):
        ux_wall = jnp.full(size, ux_wall)
    if jnp.isscalar(uy_wall):
        uy_wall = jnp.full(size, uy_wall)

    u_wall = jnp.array([ux_wall, uy_wall])
    feq_wall = get_equilibrium(rho_wall, u_wall)

    ws = WALL_SLICE[loc]
    ns = NEIGHBOR_SLICE[loc]

    if loc == "right":
        # Special case: average two interior layers for improved stability.
        rho_next, u_next = get_macroscopic(f[:, -3:-1])
        fneq_next = f[:, -3:-1] - get_equilibrium(rho_next, u_next)
        fneq_wall = jnp.mean(fneq_next, axis=1)
    else:
        rho_next, u_next = get_macroscopic(f[ns])
        fneq_wall = f[ns] - get_equilibrium(rho_next, u_next)

    f = f.at[ws].set(feq_wall + fneq_wall)
    return f


def boundary_velocity_nee(f, loc: str, ux_wall=0, uy_wall=0):
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

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """
    validate_loc(loc)
    # Density recovery uses the same formula as NEBB.
    from .nebb import _compute_wall_density
    rho_wall = _compute_wall_density(f, loc, ux_wall, uy_wall)
    return boundary_nee(f, loc, rho_wall=rho_wall, ux_wall=ux_wall, uy_wall=uy_wall)


def boundary_pressure_nee(f, loc: str, rho_wall=1):
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

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """
    validate_loc(loc)
    cfg = DENSITY_RECOVERY[loc]
    sns = SPATIAL_NEIGHBOR_SLICE[loc]

    from .nebb import _compute_wall_normal_velocity
    un_wall = _compute_wall_normal_velocity(f, loc, rho_wall)

    # Tangential velocity from neighbor node
    rho_next = jnp.sum(f[(slice(None),) + sns], axis=0)
    if cfg["normal_axis"] == 0:
        ut_wall = (jnp.sum(f[(UP_DIRS,) + sns], axis=0)
                   - jnp.sum(f[(DOWN_DIRS,) + sns], axis=0)) / rho_next
        return boundary_nee(f, loc, rho_wall=rho_wall, ux_wall=un_wall, uy_wall=ut_wall)
    else:
        ut_wall = (jnp.sum(f[(RIGHT_DIRS,) + sns], axis=0)
                   - jnp.sum(f[(LEFT_DIRS,) + sns], axis=0)) / rho_next
        return boundary_nee(f, loc, rho_wall=rho_wall, ux_wall=ut_wall, uy_wall=un_wall)
