"""D2Q9 velocity-set descriptors for 2D LBM simulations."""

from typing import NamedTuple

import numpy as np


class BoundarySpec(NamedTuple):
    """Geometry and direction metadata for a single 2D domain boundary."""

    wall: tuple
    neighbor: tuple
    in_dirs: np.ndarray
    out_dirs: np.ndarray
    tan_dirs: np.ndarray
    pos_side_dirs: np.ndarray
    neg_side_dirs: np.ndarray
    normal_sign: int
    normal_axis: int


class Lattice(NamedTuple):
    """Immutable descriptor for a 2D lattice velocity set."""

    d: int
    q: int
    w: np.ndarray
    c: np.ndarray
    cs2: float
    right_dirs: np.ndarray
    left_dirs: np.ndarray
    up_dirs: np.ndarray
    down_dirs: np.ndarray
    all_dirs: np.ndarray
    opp_dirs: np.ndarray
    boundary_spec: dict[str, BoundarySpec]


D2Q9 = Lattice(
    d=2,
    q=9,
    cs2=1.0 / 3.0,
    w=np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32),
    c=np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1],
        ],
        dtype=np.int32,
    ),
    right_dirs=np.array([1, 5, 8], dtype=np.int32),
    left_dirs=np.array([3, 7, 6], dtype=np.int32),
    up_dirs=np.array([2, 5, 6], dtype=np.int32),
    down_dirs=np.array([4, 7, 8], dtype=np.int32),
    all_dirs=np.arange(9, dtype=np.int32),
    opp_dirs=np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32),
    boundary_spec={
        "left": BoundarySpec(
            wall=(0,),
            neighbor=(1,),
            in_dirs=np.array([1, 5, 8], dtype=np.int32),
            out_dirs=np.array([3, 7, 6], dtype=np.int32),
            tan_dirs=np.array([2, 4], dtype=np.int32),
            pos_side_dirs=np.array([2, 5, 6], dtype=np.int32),
            neg_side_dirs=np.array([4, 7, 8], dtype=np.int32),
            normal_sign=1,
            normal_axis=0,
        ),
        "right": BoundarySpec(
            wall=(-1,),
            neighbor=(-2,),
            in_dirs=np.array([3, 7, 6], dtype=np.int32),
            out_dirs=np.array([1, 5, 8], dtype=np.int32),
            tan_dirs=np.array([2, 4], dtype=np.int32),
            pos_side_dirs=np.array([2, 5, 6], dtype=np.int32),
            neg_side_dirs=np.array([4, 7, 8], dtype=np.int32),
            normal_sign=-1,
            normal_axis=0,
        ),
        "top": BoundarySpec(
            wall=(slice(None), -1),
            neighbor=(slice(None), -2),
            in_dirs=np.array([4, 7, 8], dtype=np.int32),
            out_dirs=np.array([2, 5, 6], dtype=np.int32),
            tan_dirs=np.array([1, 3], dtype=np.int32),
            pos_side_dirs=np.array([1, 5, 8], dtype=np.int32),
            neg_side_dirs=np.array([3, 7, 6], dtype=np.int32),
            normal_sign=-1,
            normal_axis=1,
        ),
        "bottom": BoundarySpec(
            wall=(slice(None), 0),
            neighbor=(slice(None), 1),
            in_dirs=np.array([2, 5, 6], dtype=np.int32),
            out_dirs=np.array([4, 7, 8], dtype=np.int32),
            tan_dirs=np.array([1, 3], dtype=np.int32),
            pos_side_dirs=np.array([1, 5, 8], dtype=np.int32),
            neg_side_dirs=np.array([3, 7, 6], dtype=np.int32),
            normal_sign=1,
            normal_axis=1,
        ),
    },
)
