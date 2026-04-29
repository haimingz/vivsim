"""Multiple-relaxation-time (MRT) collision operator for the D3Q19 lattice."""

import numpy as np
import jax.numpy as jnp


M = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1],
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1],
        [0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 2, 2, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -2, -2, -2, -2],
        [0, 0, 0, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=np.float32,
)
M_INV = np.linalg.inv(M)


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
        ],
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

    return f + jnp.einsum(
        "ij,j...->i...", mrt_collision_matrix, feq - f, precision='highest'
    )
