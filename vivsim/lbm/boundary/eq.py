"""
This module implements the equilibrium boundary condition for the Lattice Boltzmann Method (LBM)
in a 2D D2Q9 lattice.

The equilibrium boundary condition assigns the distribution functions at the boundary to their
equilibrium values based on the specified density and velocity at the boundary. This method is
less accurate than the non-equilibrium bounce-back (NEBB) or non-equilibrium extrapolation (NEE)
schemes but is far more stable and easier to implement. It also supresses sound wave reflections
at the boundary.

When body forces are present, pass the corrected wall velocity obtained from
``get_corrected_wall_velocity`` (from this package) before calling these functions.
"""

from ..basic import get_equilibrium
from ._helpers import (
    BOUNDARY_SPEC,
    wrap_force_corrected,
    wrap_pressure,
    wrap_velocity,
    broadcast_wall_values,
)


def boundary_equilibrium(f, loc: str, rho_wall=1, ux_wall=0, uy_wall=0):
    """Apply the equilibrium boundary scheme with prescribed wall state.

    This is a simple boundary condition that sets the distribution functions
    at the boundary to their equilibrium values based on the specified
    density and velocity. This method is less accurate than the NEBB or NEE
    schemes but more stable.

    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        loc (str): Boundary location, one of ``'left'``, ``'right'``, ``'top'``,
            or ``'bottom'``.
        rho_wall (scalar or jax.Array of shape N): Density prescribed on the wall.
        ux_wall (scalar or jax.Array of shape N): X-velocity prescribed on the wall.
        uy_wall (scalar or jax.Array of shape N): Y-velocity prescribed on the wall.

    Returns:
        jax.Array: Updated distribution function with shape ``(9, NX, NY)``.
    """

    spec = BOUNDARY_SPEC[loc]
    
    # prepare the wall values, ensuring they are in the correct shape for broadcasting
    rho_wall, u_wall = broadcast_wall_values(
        f, loc, rho_wall=rho_wall, ux_wall=ux_wall, uy_wall=uy_wall)

    # compute the equilibrium distribution at the wall
    feq_wall = get_equilibrium(rho_wall, u_wall)

    # set the distribution at the wall to be the equilibrium distribution
    # ignoring the non-equilibrium part
    return f.at[:, *spec.wall].set(feq_wall)


boundary_velocity_equilibrium = wrap_velocity(boundary_equilibrium)
boundary_pressure_equilibrium = wrap_pressure(boundary_equilibrium)
boundary_force_corrected_equilibrium = wrap_force_corrected(boundary_equilibrium)
