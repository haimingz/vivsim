"""Boundary-condition package for the D3Q19 lattice."""

from ._helpers import (
    BOUNDARY_SPEC,
    BoundarySpec,
    broadcast_wall_values,
    get_boundary_shape,
    get_boundary_size,
    get_corrected_wall_velocity,
    get_rho_wall_from_velocity,
    get_signed_wall_velocity,
    get_wall_velocity_from_pressure,
    wrap_force_corrected,
    wrap_pressure,
    wrap_velocity,
)
from .bb import (
    boundary_bounce_back,
    boundary_specular_reflection,
    obstacle_bounce_back,
)
from .cbc import boundary_characteristic
from .eq import (
    boundary_equilibrium,
    boundary_force_corrected_equilibrium,
    boundary_pressure_equilibrium,
    boundary_velocity_equilibrium,
)
from .nebb import (
    boundary_force_corrected_nebb,
    boundary_nebb,
    boundary_pressure_nebb,
    boundary_velocity_nebb,
)
from .nee import (
    boundary_force_corrected_nee,
    boundary_nee,
    boundary_pressure_nee,
    boundary_velocity_nee,
)


__all__ = [
    "BOUNDARY_SPEC",
    "BoundarySpec",
    "broadcast_wall_values",
    "get_boundary_shape",
    "get_boundary_size",
    "get_corrected_wall_velocity",
    "get_rho_wall_from_velocity",
    "get_signed_wall_velocity",
    "get_wall_velocity_from_pressure",
    "wrap_force_corrected",
    "wrap_pressure",
    "wrap_velocity",
    "boundary_bounce_back",
    "boundary_specular_reflection",
    "obstacle_bounce_back",
    "boundary_characteristic",
    "boundary_equilibrium",
    "boundary_force_corrected_equilibrium",
    "boundary_pressure_equilibrium",
    "boundary_velocity_equilibrium",
    "boundary_nebb",
    "boundary_force_corrected_nebb",
    "boundary_pressure_nebb",
    "boundary_velocity_nebb",
    "boundary_nee",
    "boundary_force_corrected_nee",
    "boundary_pressure_nee",
    "boundary_velocity_nee",
]
