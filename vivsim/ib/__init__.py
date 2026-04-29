"""Immersed-boundary method for fluid-structure interaction (2D)."""

from .geometry import get_area, get_ds
from .kernels import kernel_cosine_4pt, kernel_peskin_3pt, kernel_peskin_4pt
from .mdf import multi_direct_forcing
from .stencil import get_ib_stencil, interpolate, spread
