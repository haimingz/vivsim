"""Immersed-boundary method for fluid-structure interaction (2D and 3D)."""

from .geometry import (
    get_area,
    get_ds_closed,
    get_ds_open,
    get_marker_da,
    get_surface_area,
    get_triangle_areas,
    get_vertex_dA,
)
from .kernels import kernel_cosine_4pt, kernel_peskin_3pt, kernel_peskin_4pt
from .mdf import multi_direct_forcing
from .stencil import (
    get_ib_stencil,
    get_ib_stencil_3d,
    interpolate,
    spread,
)

