"""Non-equilibrium bounce-back (Zou/He style) face boundary for D3Q19."""

from ..lattice import D3Q19
from ..basic import get_equilibrium
from ._helpers import (
    broadcast_wall_values,
    wrap_force_corrected,
    wrap_pressure,
    wrap_velocity,
)


def boundary_nebb(f, loc: str, rho_wall=1, ux_wall=0, uy_wall=0, uz_wall=0):
    """Apply the NEBB face boundary condition on the selected 3D boundary."""

    spec = D3Q19.boundary_spec[loc]
    out_dirs = D3Q19.opp_dirs[spec.in_dirs]

    rho_wall, u_wall = broadcast_wall_values(
        f,
        loc,
        rho_wall=rho_wall,
        ux_wall=ux_wall,
        uy_wall=uy_wall,
        uz_wall=uz_wall,
    )
    feq_wall = get_equilibrium(rho_wall, u_wall)

    wall = f[:, *spec.wall]
    new_vals = wall[out_dirs] + feq_wall[spec.in_dirs] - feq_wall[out_dirs]
    new_wall = wall.at[spec.in_dirs].set(new_vals)
    return f.at[:, *spec.wall].set(new_wall)


boundary_velocity_nebb = wrap_velocity(boundary_nebb)
boundary_pressure_nebb = wrap_pressure(boundary_nebb)
boundary_force_corrected_nebb = wrap_force_corrected(boundary_nebb)
