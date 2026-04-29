"""Equilibrium face boundary condition for the D3Q19 lattice."""

from ..lattice import D3Q19
from ..basic import get_equilibrium
from ._helpers import (
    broadcast_wall_values,
    wrap_force_corrected,
    wrap_pressure,
    wrap_velocity,
)


def boundary_equilibrium(f, loc: str, rho_wall=1, ux_wall=0, uy_wall=0, uz_wall=0):
    """Set the selected face to its prescribed equilibrium state."""

    spec = D3Q19.boundary_spec[loc]
    rho_wall, u_wall = broadcast_wall_values(
        f,
        loc,
        rho_wall=rho_wall,
        ux_wall=ux_wall,
        uy_wall=uy_wall,
        uz_wall=uz_wall,
    )
    feq_wall = get_equilibrium(rho_wall, u_wall)
    return f.at[:, *spec.wall].set(feq_wall)


boundary_velocity_equilibrium = wrap_velocity(boundary_equilibrium)
boundary_pressure_equilibrium = wrap_pressure(boundary_equilibrium)
boundary_force_corrected_equilibrium = wrap_force_corrected(boundary_equilibrium)
