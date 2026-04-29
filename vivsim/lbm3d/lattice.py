"""D3Q19 velocity-set descriptors for 3D LBM simulations.

The immutable :class:`Lattice` NamedTuple keeps D3Q19 constants and face
boundary metadata in one place. :mod:`basic` unpacks these values into
module-level names for downstream code.
"""

from typing import NamedTuple

import numpy as np


class BoundarySpec(NamedTuple):
    """Geometry and direction metadata for a single 3D domain face."""

    wall: tuple                         # slice     indices for the wall nodes
    neighbor: tuple                     # slice     indices for the neighboring nodes just inside the domain
    in_dirs: np.ndarray                 # (n_in,)   indices of directions pointing into the wall
    out_dirs: np.ndarray                # (n_out,)  indices of directions pointing out of the wall
    zero_dirs: np.ndarray               # (n_zero,) indices of directions parallel to the wall
    normal_sign: int                    # scalar    +1 if the normal vector points in the positive coordinate direction, -1 if negative
    normal_axis: int                    # scalar    the spatial axis normal to the wall, e.g. 0 for left/right faces
    tangential_axes: tuple[int, int]    # (2,)      the two spatial axes tangential to the wall, e.g. (1, 2) for left/right faces


class Lattice(NamedTuple):
    """Immutable descriptor for a 3D lattice velocity set."""

    d: int                     # scalar   spatial dimension  
    q: int                     # scalar   number of discrete velocity directions
    w: np.ndarray              # (q,)     lattice weights
    c: np.ndarray              # (q, d)   integer velocity vectors
    cs2: float                 # scalar   speed of sound squared
    all_dirs: np.ndarray       # (q,)     indices of all directions
    opp_dirs: np.ndarray       # (q,)     index of the opposite direction
    boundary_spec: dict[str, BoundarySpec]


D3Q19 = Lattice(
    d=3,
    q=19,
    cs2=1.0 / 3.0,
    w=np.array([
        1/3,
        1/18, 1/18, 1/18, 1/18, 1/18, 1/18,
        1/36, 1/36, 1/36, 1/36,
        1/36, 1/36, 1/36, 1/36,
        1/36, 1/36, 1/36, 1/36,
    ], dtype=np.float32),
    c=np.array([
        [ 0,  0,  0],
        [ 1,  0,  0], [-1,  0,  0],
        [ 0,  1,  0], [ 0, -1,  0],
        [ 0,  0,  1], [ 0,  0, -1],
        [ 1,  1,  0], [-1,  1,  0], [ 1, -1,  0], [-1, -1,  0],
        [ 1,  0,  1], [-1,  0,  1], [ 1,  0, -1], [-1,  0, -1],
        [ 0,  1,  1], [ 0, -1,  1], [ 0,  1, -1], [ 0, -1, -1],
    ], dtype=np.int32),
    all_dirs=np.arange(19, dtype=np.int32),
    opp_dirs=np.array(
        [0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15],
        dtype=np.int32,
    ),
    boundary_spec={
        "left": BoundarySpec(
            wall=(0, slice(None), slice(None)),
            neighbor=(1, slice(None), slice(None)),
            in_dirs=np.array([1, 7, 9, 11, 13], dtype=np.int32),
            out_dirs=np.array([2, 8, 10, 12, 14], dtype=np.int32),
            zero_dirs=np.array([0, 3, 4, 5, 6, 15, 16, 17, 18], dtype=np.int32),
            tangential_axes=(1, 2),
            normal_sign=1,
            normal_axis=0,
        ),
        "right": BoundarySpec(
            wall=(-1, slice(None), slice(None)),
            neighbor=(-2, slice(None), slice(None)),
            in_dirs=np.array([2, 8, 10, 12, 14], dtype=np.int32),
            out_dirs=np.array([1, 7, 9, 11, 13], dtype=np.int32),
            zero_dirs=np.array([0, 3, 4, 5, 6, 15, 16, 17, 18], dtype=np.int32),
            tangential_axes=(1, 2),
            normal_sign=-1,
            normal_axis=0,
        ),
        "bottom": BoundarySpec(
            wall=(slice(None), 0, slice(None)),
            neighbor=(slice(None), 1, slice(None)),
            in_dirs=np.array([3, 7, 8, 15, 17], dtype=np.int32),
            out_dirs=np.array([4, 9, 10, 16, 18], dtype=np.int32),
            zero_dirs=np.array([0, 1, 2, 5, 6, 11, 12, 13, 14], dtype=np.int32),
            tangential_axes=(0, 2),
            normal_sign=1,
            normal_axis=1,
        ),
        "top": BoundarySpec(
            wall=(slice(None), -1, slice(None)),
            neighbor=(slice(None), -2, slice(None)),
            in_dirs=np.array([4, 9, 10, 16, 18], dtype=np.int32),
            out_dirs=np.array([3, 7, 8, 15, 17], dtype=np.int32),
            zero_dirs=np.array([0, 1, 2, 5, 6, 11, 12, 13, 14], dtype=np.int32),
            tangential_axes=(0, 2),
            normal_sign=-1,
            normal_axis=1,
        ),
        "back": BoundarySpec(
            wall=(slice(None), slice(None), 0),
            neighbor=(slice(None), slice(None), 1),
            in_dirs=np.array([5, 11, 12, 15, 16], dtype=np.int32),
            out_dirs=np.array([6, 13, 14, 17, 18], dtype=np.int32),
            zero_dirs=np.array([0, 1, 2, 3, 4, 7, 8, 9, 10], dtype=np.int32),
            tangential_axes=(0, 1),
            normal_sign=1,
            normal_axis=2,
        ),
        "front": BoundarySpec(
            wall=(slice(None), slice(None), -1),
            neighbor=(slice(None), slice(None), -2),
            in_dirs=np.array([6, 13, 14, 17, 18], dtype=np.int32),
            out_dirs=np.array([5, 11, 12, 15, 16], dtype=np.int32),
            zero_dirs=np.array([0, 1, 2, 3, 4, 7, 8, 9, 10], dtype=np.int32),
            tangential_axes=(0, 1),
            normal_sign=-1,
            normal_axis=2,
        ),
    },
)
