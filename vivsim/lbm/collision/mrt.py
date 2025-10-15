"""
This file implements the multiple relaxation time (MRT) collision operator.
MRT improves numerical stability by relaxing different moments at different rates.
"""

import jax.numpy as jnp


# Transformation matrix (Gram-Schmidt method) 
M = jnp.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [-4, -1, -1, -1, -1, 2, 2, 2, 2],
            [4, -2, -2, -2, -2, 1, 1, 1, 1],
            [0, 1, 0, -1, 0, 1, -1, -1, 1],
            [0, -2, 0, 2, 0, 1, -1, -1, 1],
            [0, 0, 1, 0, -1, 1, 1, -1, -1],
            [0, 0, -2, 0, 2, 1, 1, -1, -1],
            [0, 1, -1, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, 1, -1],
        ]
    )

# Inverse of the transformation matrix (precomputed for efficiency)
M_INV = jnp.linalg.inv(M)


def get_mrt_relaxation_matrix(omega):
    """Construct the relaxation matrix S for the MRT collision model.
    
    The relaxation matrix contains different relaxation rates for different moments.
    Conserved moments (mass and momentum) have zero relaxation, while non-conserved
    moments are relaxed at different rates to improve numerical stability.
    
    Args:
        omega (float): Relaxation parameter for the shear viscosity modes. This value
            determines the kinematic viscosity: nu = (1/omega - 0.5) / 3.
    
    Returns:
        jax.Array: Diagonal relaxation matrix S with shape (9, 9). The diagonal elements
            correspond to relaxation rates for: [density, energy, energy², momentum_x,
            energy flux_x, momentum_y, energy flux_y, stress_xx, stress_xy].
    """
    return jnp.diag(jnp.array([0, 1.4, 1.4, 0, 1.2, 0, 1.2, omega, omega]))


def get_mrt_collision_operator(omega):
    """Pre-compute the constant MRT collision operator matrix.
    
    This function computes the complete MRT collision operator M^{-1} @ S @ M, which
    transforms from velocity space to moment space, applies relaxation, and transforms
    back. Pre-computing this matrix improves computational efficiency.
    
    Args:
        omega (float): Relaxation parameter for the shear viscosity modes. This value
            determines the kinematic viscosity: nu = (1/omega - 0.5) / 3.
    
    Returns:
        jax.Array: MRT collision operator matrix with shape (9, 9), representing the
            transformation M^{-1} @ S @ M.
    """
    S = get_mrt_relaxation_matrix(omega)
    return M_INV @ S @ M


def collision_mrt(f, feq, mrt_collision_matrix):
    """Perform the collision step using the Multiple Relaxation Time (MRT) model.
    
    The MRT collision operator relaxes different moments at different rates, providing
    better numerical stability compared to the single relaxation time (BGK) model. The
    collision is computed as f_new = f - M^{-1} @ S @ M @ (f - f^eq).
    
    Args:
        f (jax.Array): Current distribution function with shape (9, NX, NY).
        feq (jax.Array): Equilibrium distribution function with shape (9, NX, NY).
        mrt_collision_matrix (jax.Array): Pre-computed MRT collision operator matrix
            with shape (9, 9), obtained from `get_mrt_collision_operator`. This
            represents M^{-1} @ S @ M.
    
    Returns:
        jax.Array: Post-collision distribution function with shape (9, NX, NY).
    
    Note:
        The MRT collision matrix should be pre-computed using `get_mrt_collision_operator`
        and passed as an argument to avoid recomputing it at each time step.
    """
    
    return f + jnp.einsum('ij,jxy->ixy', mrt_collision_matrix, feq - f)
