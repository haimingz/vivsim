"""This file implements the multiple relaxation time (MRT) collision model 
for the lattice Boltzmann method (LBM)."""

import jax.numpy as jnp


# ---------- Transformation matrix (Gram-Schmidt method) ----------

def get_trans_matrix():
    """Get the transformation matrix for the MRT collision model."""
    
    return jnp.array(
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

# ----------------- collision term -----------------

def get_relax_matrix(omega):
    """Get the relaxation matrix for the MRT collision model
    according to the given relaxation parameter omega."""

    return jnp.diag(jnp.array([1, 1.4, 1.4, 1, 1.2, 1, 1.2, omega, omega]))

def get_collision_left_matrix(M, S):
    """precomputed the constant left matrix for the MRT collision model.
    
    Args:
        M: Transformation matrix, shape (9,9)
        S: Relaxation matrix, shape (9,9)
    Returns:
        Left matrix (M^-1 @ S @ M) for the MRT collision model, shape (9,9)
    """
    
    return jnp.linalg.inv(M) @ S @ M

def collision(f, feq, left_matrix):
    """Perform the collision step using the MRT model.
    
    The left matrix (M^-1 @ S @ M) can be pre-computed using `get_collision_left_matrix`
    and passed as an argument to avoid recomputing it at each time step.
    """
    
    return jnp.tensordot(left_matrix, feq - f, axes=([1], [0])) + f

# ----------------- forcing term (Guo forcing) -----------------

def get_source_left_matrix(M, S):
    """Pre-compute the constant left matrix for the MRT source term.

    Args:
        M: Transformation matrix, shape (9,9)
        S: Relaxation matrix, shape (9,9)
    Returns:
        Left matrix (M^-1 @ (I - 0.5 * S) @ M), shape (9,9)
    """
    
    return jnp.linalg.inv(M) @ (jnp.eye(9) - 0.5 * S) @ M



def get_source(gd, left_matrix):
    """Compute the source term according to Guo's scheme.
    
    The left matrix (M^-1 @ (I - 0.5 * S) @ M) can be pre-computed using `get_source_left_matrix`
    and passed as an argument to avoid recomputing it at each time step.
    """
    
    return jnp.tensordot(left_matrix, gd, axes=([1], [0]))