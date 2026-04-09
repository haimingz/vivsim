"""Non-equilibrium extrapolation (NEE) face boundary for the D3Q19 lattice."""

from ..basic import get_equilibrium, get_macroscopic
from ._helpers import (
    BOUNDARY_SPEC,
    broadcast_wall_values,
    wrap_force_corrected,
    wrap_pressure,
    wrap_velocity,
)


def boundary_nee(f, loc: str, rho_wall=1, ux_wall=0, uy_wall=0, uz_wall=0):
    """Apply the NEE face boundary condition on the selected 3D boundary."""

    spec = BOUNDARY_SPEC[loc]
    rho_wall, u_wall = broadcast_wall_values(
        f,
        loc,
        rho_wall=rho_wall,
        ux_wall=ux_wall,
        uy_wall=uy_wall,
        uz_wall=uz_wall,
    )
    feq_wall = get_equilibrium(rho_wall, u_wall)

    f_neighbor = f[:, *spec.neighbor]
    rho_neighbor, u_neighbor = get_macroscopic(f_neighbor)
    feq_neighbor = get_equilibrium(rho_neighbor, u_neighbor)
    fneq_neighbor = f_neighbor - feq_neighbor

    return f.at[:, *spec.wall].set(feq_wall + fneq_neighbor)


boundary_velocity_nee = wrap_velocity(boundary_nee)
boundary_pressure_nee = wrap_pressure(boundary_nee)
boundary_force_corrected_nee = wrap_force_corrected(boundary_nee)
