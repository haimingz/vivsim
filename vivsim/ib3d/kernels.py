"""3D immersed-boundary kernel wrappers.

These kernels are dimension-independent, so the 3D package keeps the same API
but delegates the implementation to the 2D module to avoid duplicate logic.
"""

from ..ib.kernels import (
    kernel_cosine_4pt as _kernel_cosine_4pt,
    kernel_peskin_3pt as _kernel_peskin_3pt,
    kernel_peskin_4pt as _kernel_peskin_4pt,
)


def kernel_peskin_3pt(r):
    """Return the 3-point discrete delta kernel."""

    return _kernel_peskin_3pt(r)


def kernel_peskin_4pt(r):
    """Return the 4-point discrete delta kernel."""

    return _kernel_peskin_4pt(r)


def kernel_cosine_4pt(r):
    """Return the 4-point cosine delta kernel."""

    return _kernel_cosine_4pt(r)
