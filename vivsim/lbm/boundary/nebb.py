"""
This module implements the Non-Equilibrium Bounce-Back (or Zou/He) boundary condition
for the Lattice Boltzmann Method (LBM) in a 2D D2Q9 lattice.

When body forces are present, pass the corrected wall velocity obtained from
``get_corrected_wall_velocity`` (from this package) before calling these functions.
"""

from ..basic import LEFT_DIRS, RIGHT_DIRS, UP_DIRS, DOWN_DIRS
import jax.numpy as jnp
from ._slices import (
    WALL_SLICE, NEIGHBOR_SLICE,
    SPATIAL_WALL_SLICE, SPATIAL_NEIGHBOR_SLICE,
    BOUNDARY_SIZE_AXIS,
    NEBB_CONFIG, DENSITY_RECOVERY, validate_loc,
)


def boundary_nebb(f, loc: str, rho_wall=1, ux_wall=0, uy_wall=0):
    """Non-Equilibrium Bounce-Back (or Zou/He) scheme.

    This is the core function for the nebb_velocity and nebb_pressure functions.

    It applies the Non-Equilibrium Bounce-Back (or Zou/He) scheme to the distribution function
    at the specified boundary location, enforcing the given wall velocity and density.
    It only modifies the unknown incoming distribution functions at the boundary, while keeping
    the known outgoing distribution functions unchanged.

    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        loc (str): The boundary where the velocity condition is enforced,
            can be 'left', 'right', 'top', or 'bottom'.
        ux_wall (scalar or jax.Array of shape NX or NY): The x-component of velocity.
        uy_wall (scalar or jax.Array of shape NX or NY): The y-component of velocity.
        rho_wall (scalar or jax.Array of shape NX or NY): The density at the wall.

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """
    validate_loc(loc)
    cfg = NEBB_CONFIG[loc]
    sws = SPATIAL_WALL_SLICE[loc]
    i0, i1, i2 = cfg["incoming"]
    s0, s1, s2 = cfg["source"]
    t0, t1 = cfg["tangential"]
    sn = cfg["normal_sign"]
    un = ux_wall if cfg["normal_axis"] == 0 else uy_wall
    ut = uy_wall if cfg["normal_axis"] == 0 else ux_wall

    f = f.at[(i0,) + sws].set(
        f[(s0,) + sws] + sn * 2 / 3 * un * rho_wall
    )
    f = f.at[(i1,) + sws].set(
        f[(s1,) + sws]
        - sn * 0.5 * (f[(t0,) + sws] - f[(t1,) + sws])
        + sn * (1 / 6 * un + 0.5 * ut) * rho_wall
    )
    f = f.at[(i2,) + sws].set(
        f[(s2,) + sws]
        + sn * 0.5 * (f[(t0,) + sws] - f[(t1,) + sws])
        + sn * (1 / 6 * un - 0.5 * ut) * rho_wall
    )
    return f


def _compute_wall_density(f, loc, ux_wall=0, uy_wall=0):
    """Compute wall density from known outgoing populations (for velocity BCs)."""
    cfg = DENSITY_RECOVERY[loc]
    sws = SPATIAL_WALL_SLICE[loc]
    un = ux_wall if cfg["normal_axis"] == 0 else uy_wall
    zero_sum = sum(f[(d,) + sws] for d in cfg["zero"])
    out_sum = sum(f[(d,) + sws] for d in cfg["outgoing"])
    return (zero_sum + 2 * out_sum) / (1 + cfg["sign"] * un)


def _compute_wall_normal_velocity(f, loc, rho_wall):
    """Compute normal wall velocity from known populations (for pressure BCs)."""
    cfg = DENSITY_RECOVERY[loc]
    sws = SPATIAL_WALL_SLICE[loc]
    zero_sum = sum(f[(d,) + sws] for d in cfg["zero"])
    out_sum = sum(f[(d,) + sws] for d in cfg["outgoing"])
    return (zero_sum + 2 * out_sum) / rho_wall - 1


def boundary_velocity_nebb(f, loc: str, ux_wall=0, uy_wall=0):
    """Enforce given velocity ux_wall, uy_wall at the specified boundary using the
    Non-Equilibrium Bounce-Back (or Zou/He) scheme.

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
    rho_wall = _compute_wall_density(f, loc, ux_wall, uy_wall)
    return boundary_nebb(f, loc, ux_wall=ux_wall, uy_wall=uy_wall, rho_wall=rho_wall)


def boundary_pressure_nebb(f, loc: str, rho_wall=1):
    """Enforce given pressure (density) rho_wall at the specified boundary using the
    Non-Equilibrium Bounce-Back (or Zou/He) scheme.

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

    un_wall = _compute_wall_normal_velocity(f, loc, rho_wall)

    # Tangential velocity from neighbor node
    rho_next = jnp.sum(f[(slice(None),) + sns], axis=0)
    if cfg["normal_axis"] == 0:
        ut_wall = (jnp.sum(f[(UP_DIRS,) + sns], axis=0)
                   - jnp.sum(f[(DOWN_DIRS,) + sns], axis=0)) / rho_next
        return boundary_nebb(f, loc, ux_wall=un_wall, uy_wall=ut_wall, rho_wall=rho_wall)
    else:
        ut_wall = (jnp.sum(f[(RIGHT_DIRS,) + sns], axis=0)
                   - jnp.sum(f[(LEFT_DIRS,) + sns], axis=0)) / rho_next
        return boundary_nebb(f, loc, ux_wall=ut_wall, uy_wall=un_wall, rho_wall=rho_wall)
