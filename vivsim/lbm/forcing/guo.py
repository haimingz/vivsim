import jax.numpy as jnp
from ..basic import VELOCITIES, WEIGHTS
from ..collision.mrt import M, M_INV, get_mrt_relaxation_matrix


def get_guo_forcing_term(g, u):
    """Compute the forcing term using Guo's forcing scheme.
    
    This function discretizes the external force density into the lattice Boltzmann
    framework using Guo's forcing scheme. 
    
    Args:
        g (jax.Array): External force density with shape (2, NX, NY), where the first
            dimension represents force components in x and y directions.
        u (jax.Array): Fluid velocity field with shape (2, NX, NY), where the first
            dimension represents velocity components in x and y directions.
    
    Returns:
        jax.Array: Lattice forcing term with shape (9, NX, NY), where 9 is the number
            of discrete velocity directions in the D2Q9 lattice.
    """
    uc = (u[0, :, :] * VELOCITIES[:, 0, jnp.newaxis, jnp.newaxis] + 
          u[1, :, :] * VELOCITIES[:, 1, jnp.newaxis, jnp.newaxis])
    
    g_lattice = WEIGHTS[..., jnp.newaxis, jnp.newaxis] * (
        g[0] * (
            3 * (VELOCITIES[:, 0, jnp.newaxis, jnp.newaxis] - u[jnp.newaxis, 0,...]) 
            + 9 * (uc * VELOCITIES[:,0, jnp.newaxis, jnp.newaxis])) 
        + g[1] * (
            3 * (VELOCITIES[:, 1, jnp.newaxis, jnp.newaxis] - u[jnp.newaxis, 1,...]) 
            + 9 * (uc * VELOCITIES[:, 1, jnp.newaxis, jnp.newaxis])))
    
    return g_lattice


# ----------------- for srt -----------------

def forcing_guo_bgk(f, g, u, omega):
    """Apply Guo's forcing scheme for BGK (single relaxation time) collision operator.
    
    This function applies the external force using Guo's forcing scheme, which is
    specifically designed for the BGK collision model. The forcing term is scaled
    by (1 - 0.5 * omega) to ensure correct hydrodynamic equations.
    
    Args:
        f (jax.Array): Distribution function with shape (9, NX, NY).
        g (jax.Array): External force density with shape (2, NX, NY), where the first
            dimension represents force components in x and y directions.
        u (jax.Array): Fluid velocity field with shape (2, NX, NY), where the first
            dimension represents velocity components in x and y directions.
        omega (float): BGK relaxation parameter (inverse of relaxation time).
    
    Returns:
        jax.Array: Updated distribution function with applied forcing term, shape (9, NX, NY).    
    """
    g_lattice = get_guo_forcing_term(g, u)    
    return f + g_lattice * (1 - 0.5 * omega)


# ----------------- for mrt -----------------

def get_mrt_forcing_operator(omega):
    """Pre-compute the constant forcing operator matrix for MRT collision.

    This function computes the transformation operator M^{-1} @ (I - 0.5 * S) @ M,
    which is used to apply the forcing term in moment space for the Multiple
    Relaxation Time (MRT) collision operator. Pre-computing this matrix improves
    computational efficiency.

    Args:
        omega (float): BGK relaxation parameter used to construct the MRT relaxation
            matrix S. This determines the relaxation rates for different moments.
    
    Returns:
        jax.Array: MRT forcing operator matrix with shape (9, 9), representing the
            transformation M^{-1} @ (I - 0.5 * S) @ M.
    """
    S = get_mrt_relaxation_matrix(omega)
    return M_INV @ (jnp.eye(9) - 0.5 * S) @ M


def forcing_guo_mrt(f, g, u, mrt_forcing_operator):
    """Apply Guo's forcing scheme for MRT (multiple relaxation time) collision operator.
    
    This function applies the external force using Guo's forcing scheme adapted for
    the MRT collision model. The forcing is applied in moment space using a
    pre-computed transformation matrix.
    
    Args:
        f (jax.Array): Distribution function with shape (9, NX, NY).
        g (jax.Array): External force density with shape (2, NX, NY), where the first
            dimension represents force components in x and y directions.
        u (jax.Array): Fluid velocity field with shape (2, NX, NY), where the first
            dimension represents velocity components in x and y directions.
        mrt_forcing_operator (jax.Array): Pre-computed MRT forcing operator matrix with
            shape (9, 9), obtained from `get_mrt_forcing_operator`. This represents
            M^{-1} @ (I - 0.5 * S) @ M.
    
    Returns:
        jax.Array: Updated distribution function with applied forcing term, shape (9, NX, NY).
    
    Note:
        The MRT forcing operator should be pre-computed using `get_mrt_forcing_operator`
        and passed as an argument to avoid recomputing it at each time step.
    """
    g_lattice = get_guo_forcing_term(g, u)   
    return f + jnp.tensordot(mrt_forcing_operator, g_lattice, axes=([1], [0]))
