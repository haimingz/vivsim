"""Immersed-boundary method for fluid-structure interaction."""

from .geometry import get_area, get_ds_closed, get_ds_open
from .kernels import kernel_peskin_3pt, kernel_peskin_4pt, kernel_cosine_4pt
from .mdf import multi_direct_forcing
from .stencil import (
    get_ib_stencil,
    interpolate,
    spread,
)

