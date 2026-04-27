"""Immersed-boundary method for fluid-structure interaction (3D)."""

from .geometry import get_surface_area, get_triangle_areas, get_vertex_dA
from .stencil import get_ib_stencil
from vivsim.ib.kernels import kernel_cosine_4pt, kernel_peskin_3pt, kernel_peskin_4pt
from vivsim.ib.mdf import multi_direct_forcing
from vivsim.ib.stencil import interpolate, spread
