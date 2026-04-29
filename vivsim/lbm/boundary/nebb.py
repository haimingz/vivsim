"""
This module implements the Non-Equilibrium Bounce-Back (or Zou/He) boundary condition
for the Lattice Boltzmann Method (LBM) in a 2D D2Q9 lattice.

When body forces are present, pass the corrected wall velocity obtained from
``get_corrected_wall_velocity`` (from this package) before calling these functions.
"""

import jax.numpy as jnp

from ._helpers import (
    BOUNDARY_SPEC,
    get_signed_wall_velocity,
    wrap_force_corrected,
    wrap_pressure,
    wrap_velocity,
)


def boundary_nebb(f, loc: str, rho_wall=1, ux_wall=0, uy_wall=0):
    """Non-Equilibrium Bounce-Back (or Zou/He) scheme.

    This is the core function for the velocity, pressure, and force-corrected
    NEBB wrappers defined at the bottom of this module.

    It applies the Non-Equilibrium Bounce-Back (or Zou/He) scheme to the distribution function
    at the specified boundary location, enforcing the given wall velocity and density.
    It only modifies the unknown incoming distribution functions at the boundary, while keeping
    the known outgoing distribution functions unchanged.

    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        loc (str): Boundary location, one of ``'left'``, ``'right'``, ``'top'``,
            or ``'bottom'``.
        ux_wall (scalar or jax.Array of shape N): The x-component of velocity.
        uy_wall (scalar or jax.Array of shape N): The y-component of velocity.
        rho_wall (scalar or jax.Array of shape N): The density at the wall.

    Returns:
        jax.Array: Updated distribution function with shape ``(9, NX, NY)``.
    """

    spec = BOUNDARY_SPEC[loc]
    un, ut = get_signed_wall_velocity(loc, ux_wall=ux_wall, uy_wall=uy_wall)

    wall = f[:, *spec.wall]  # shape (9, N) – cheap slice view

    shear_term = 0.5 * (wall[spec.tan_dirs[0]] - wall[spec.tan_dirs[1]]) * spec.normal_sign
    normal_term = 1 / 6 * un * rho_wall
    tangential_term = 0.5 * ut * rho_wall

    new_vals = jnp.stack([
        wall[spec.out_dirs[0]] + 2 / 3 * un * rho_wall,
        wall[spec.out_dirs[1]] - shear_term + normal_term + tangential_term,
        wall[spec.out_dirs[2]] + shear_term + normal_term - tangential_term,
    ])  # (3, N)
    new_wall = wall.at[jnp.array(spec.in_dirs)].set(new_vals)  # update 3 rows in (9, N)
    return f.at[:, *spec.wall].set(new_wall)  # single full-buffer copy

boundary_velocity_nebb = wrap_velocity(boundary_nebb)
boundary_pressure_nebb = wrap_pressure(boundary_nebb)
boundary_force_corrected_nebb = wrap_force_corrected(boundary_nebb)
