"""Multiple-relaxation-time (MRT) collision operator for the D3Q19 lattice."""

import numpy as np
import jax.numpy as jnp

from ..basic import Q, VELOCITIES


CONSERVED_ROWS = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
BULK_ROW = 4
SHEAR_ROWS = jnp.array([5, 6, 7, 8, 9], dtype=jnp.int32)
GHOST_ROWS = jnp.arange(10, Q, dtype=jnp.int32)


def _build_moment_matrix():
    """Build an invertible polynomial moment basis for D3Q19.

    The first four rows are the conserved moments ``[1, cx, cy, cz]``.
    Rows 4-9 are second-order moments, with rows 5-9 carrying the viscous
    stress modes that should relax with ``omega``.
    """

    c = np.asarray(VELOCITIES, dtype=np.float64)
    cx = c[:, 0]
    cy = c[:, 1]
    cz = c[:, 2]

    cx2 = cx**2
    cy2 = cy**2
    cz2 = cz**2

    rows = np.array(
        [
            np.ones(Q),
            cx,
            cy,
            cz,
            cx2 + cy2 + cz2,
            2.0 * cx2 - cy2 - cz2,
            cy2 - cz2,
            cx * cy,
            cx * cz,
            cy * cz,
            cx2 * cy,
            cx2 * cz,
            cy2 * cx,
            cy2 * cz,
            cz2 * cx,
            cz2 * cy,
            cx2 * cy2,
            cx2 * cz2,
            cy2 * cz2,
        ],
        dtype=np.float64,
    )

    if np.linalg.matrix_rank(rows) != Q:
        raise ValueError("Failed to construct a full-rank D3Q19 MRT moment matrix.")

    return rows


_M = _build_moment_matrix()
M = jnp.asarray(_M)
M_INV = jnp.asarray(np.linalg.inv(_M))


def get_mrt_relaxation_matrix(omega):
    """Construct the D3Q19 diagonal relaxation matrix.

    The relaxation rates are grouped as follows:
    - rows 0-3: conserved moments, fixed to zero
    - row 4: bulk / trace second-order mode
    - rows 5-9: viscous stress modes, relaxed with ``omega``
    - rows 10-18: higher-order ghost modes

    Args:
        omega (float): Relaxation parameter for the shear-viscosity modes.

    Returns:
        jax.Array: Diagonal relaxation matrix with shape ``(19, 19)``.
    """

    s = jnp.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.1,
            omega,
            omega,
            omega,
            omega,
            omega,
            1.2,
            1.2,
            1.2,
            1.2,
            1.2,
            1.2,
            1.4,
            1.4,
            1.4,
        ]
    )
    return jnp.diag(s)


def get_mrt_collision_operator(omega):
    """Pre-compute the constant D3Q19 MRT collision operator matrix."""

    S = get_mrt_relaxation_matrix(omega)
    return M_INV @ S @ M


def collision_mrt(f, feq, mrt_collision_matrix):
    """Perform the MRT collision step for D3Q19.

    Args:
        f (jax.Array): Distribution function with shape ``(19, *spatial_dims)``.
        feq (jax.Array): Equilibrium distribution with the same shape as ``f``.
        mrt_collision_matrix (jax.Array): Pre-computed MRT operator with shape
            ``(19, 19)``.

    Returns:
        jax.Array: Post-collision distribution function.
    """

    return f + jnp.einsum("ij,j...->i...", mrt_collision_matrix, feq - f)
