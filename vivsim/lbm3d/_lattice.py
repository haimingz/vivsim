"""Lattice velocity-set descriptors for 3D LBM simulations.

Each lattice is an immutable :class:`Lattice` NamedTuple that bundles all
constants for a given model.  :mod:`basic` unpacks the active lattice into
module-level names so that all downstream code is unaffected when the model
changes.

To add D3Q27: define a new ``Lattice(...)`` literal below and swap the
import in :mod:`basic`.
"""

from typing import NamedTuple

import numpy as np


class Lattice(NamedTuple):
    """Immutable descriptor for a 3D lattice velocity set."""

    dim: int
    q: int
    cs2: float
    weights: np.ndarray      # (q,)     lattice weights
    velocities: np.ndarray   # (q, dim) integer velocity vectors
    opp_dirs: np.ndarray     # (q,)     index of the opposite direction
    right_dirs: np.ndarray   # velocities[:, 0] == +1
    left_dirs: np.ndarray    # velocities[:, 0] == -1
    up_dirs: np.ndarray      # velocities[:, 1] == +1
    down_dirs: np.ndarray    # velocities[:, 1] == -1
    front_dirs: np.ndarray   # velocities[:, 2] == +1
    back_dirs: np.ndarray    # velocities[:, 2] == -1


D3Q19 = Lattice(
    dim=3,
    q=19,
    cs2=1.0 / 3.0,
    weights=np.array([
        1/3,
        1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
        1/36, 1/36, 1/36, 1/36,
        1/36, 1/36, 1/36, 1/36,
        1/36, 1/36, 1/36, 1/36,
    ]),
    velocities=np.array([
        [ 0,  0,  0],
        [ 1,  0,  0], [-1,  0,  0],
        [ 0,  1,  0], [ 0, -1,  0],
        [ 0,  0,  1], [ 0,  0, -1],
        [ 1,  1,  0], [-1,  1,  0], [ 1, -1,  0], [-1, -1,  0],
        [ 1,  0,  1], [-1,  0,  1], [ 1,  0, -1], [-1,  0, -1],
        [ 0,  1,  1], [ 0, -1,  1], [ 0,  1, -1], [ 0, -1, -1],
    ], dtype=np.int32),
    opp_dirs=np.array(
        [0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15],
        dtype=np.int32,
    ),
    right_dirs=np.array([ 1,  7,  9, 11, 13]),
    left_dirs =np.array([ 2,  8, 10, 12, 14]),
    up_dirs   =np.array([ 3,  7,  8, 15, 17]),
    down_dirs =np.array([ 4,  9, 10, 16, 18]),
    front_dirs=np.array([ 5, 11, 12, 15, 16]),
    back_dirs =np.array([ 6, 13, 14, 17, 18]),
)
