"""Immersed-boundary method for fluid-structure interaction."""

from .geometry import get_area as get_area, get_ds_closed as get_ds_closed, get_ds_open as get_ds_open
from .kernels import kernel_peskin_3pt as kernel_peskin_3pt, kernel_peskin_4pt as kernel_peskin_4pt, kernel_cosine_4pt as kernel_cosine_4pt
from .mdf import multi_direct_forcing as multi_direct_forcing
from .stencil import (
    get_ib_stencil as get_ib_stencil,
    interpolate as interpolate,
    spread as spread,
)
