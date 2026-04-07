"""
This module implements the Non-Equilibrium Extrapolation (NEE) boundary condition
for the Lattice Boltzmann Method (LBM) in a 2D D2Q9 lattice.

When body forces are present, pass the corrected wall velocity obtained from
``get_corrected_wall_velocity`` (from this package) before calling these functions.
"""

import jax.numpy as jnp
from ..basic import get_equilibrium, get_macroscopic
from ._helpers import (
    BOUNDARY_SPEC,
    broadcast_wall_values,
    wrap_force_corrected,
    wrap_pressure,
    wrap_velocity,
)


def boundary_nee(f, loc: str, rho_wall=1, ux_wall=0, uy_wall=0):
    """Non-Equilibrium Extrapolation scheme.

    This is the core function for the velocity, pressure, and force-corrected
    NEE wrappers defined at the bottom of this module.

    It applies the Non-Equilibrium Extrapolation scheme to the distribution function
    at the specified boundary location, enforcing the given wall velocity and density.
    It modifies all the distribution functions at the boundary in such a way that
    ensures a zero-gradient condition for the non-equilibrium part of the
    distribution functions at the boundary.

    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        loc (str): Boundary location, one of ``'left'``, ``'right'``, ``'top'``,
            or ``'bottom'``.
        rho_wall (scalar or jax.Array of shape N): The density at the wall.
        ux_wall (scalar or jax.Array of shape N): The x-component of velocity.
        uy_wall (scalar or jax.Array of shape N): The y-component of velocity.

    Returns:
        jax.Array: Updated distribution function with shape ``(9, NX, NY)``.
    """

    spec = BOUNDARY_SPEC[loc]

    # prepare the wall values, ensuring they are in the correct shape for broadcasting
    rho_wall, u_wall = broadcast_wall_values(
        f, loc, rho_wall=rho_wall, ux_wall=ux_wall, uy_wall=uy_wall)
    
    # compute the equilibrium distribution at the wall
    feq_wall = get_equilibrium(rho_wall, u_wall)

    # compute the non-equilibrium part of the distribution at the neighboring fluid node
    f_neighbor = f[:, *spec.neighbor]
    rho_neighbor, u_neighbor = get_macroscopic(f_neighbor)
    feq_neighbor = get_equilibrium(rho_neighbor, u_neighbor)
    fneq_neighbor = f_neighbor - feq_neighbor

    # set the wall state to equilibrium at the wall plus the extrapolated
    # non-equilibrium contribution from the adjacent fluid node
    return f.at[:, *spec.wall].set(feq_wall + fneq_neighbor)


boundary_velocity_nee = wrap_velocity(boundary_nee)
boundary_pressure_nee = wrap_pressure(boundary_nee)
boundary_force_corrected_nee = wrap_force_corrected(boundary_nee)
