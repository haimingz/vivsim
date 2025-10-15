"""
This module implements the Karlin–Bösch–Chikatamarla (KBC) collision operator.
The KBC model is based on the entropic principle to ensure numerical stability.
"""


import jax.numpy as jnp


def collision_kbc(f: jnp.ndarray, feq: jnp.ndarray, omega: float) -> jnp.ndarray:
    """
    Karlin–Bösch–Chikatamarla (KBC) collision operator for LBM.
    Based on the entropic principle to ensure numerical stability.

    Args:
        f (jnp.ndarray): Current discrete distribution function (DDF).
        feq (jnp.ndarray): Equilibrium DDF.
        omega (float): Relaxation parameter (= 1 / relaxation time).

    Returns:
        jnp.ndarray: Post-collision DDF (f_post).
        
    """

    # Nonequilibrium distribution
    fneq = f - feq

    # shear component of the non-equilibrium distribution
    
    # Step 1:
    # Non-equilibrium momentum flux tensor (\Pi^neq) is the second order moment of fneq
    # calculated by $\Pi^neq_{\alpha\beta} = \sum_i f_i^neq c_{i\alpha} c_{i\beta}$
    # normal stress part: $\Pi^neq_{xx} = \sum_i f_i^neq c_{ix} c_{ix}$ and $\Pi^neq_{yy} = \sum_i f_i^neq c_{iy} c_{iy}$
    # shear stress part: $\Pi^neq_{xy} = \sum_i f_i^neq c_{ix} c_{iy}$
    # normal stress difference = \Pi^neq_{xx} - \Pi^neq_{yy}
    # For D2Q9, substituting the discrete velocities, we have:
    normal_stress_diff = fneq[1, ...] - fneq[2, ...] + fneq[3, ...] - fneq[4, ...]
    shear_stress = fneq[5, ...] - fneq[6, ...] + fneq[7, ...] - fneq[8, ...]
    
    # Step 2: calculate the shear part of the fneq
    signs = jnp.array([1, -1, 1, -1])[:, None, None]
    shear_part = jnp.zeros_like(fneq)
    shear_part = shear_part.at[1:5, ...].set(signs * normal_stress_diff / 4.0)
    shear_part = shear_part.at[5:9, ...].set(signs * shear_stress / 4.0)    

    # Higher-order (non-hydrodynamic) component of the fneq
    high_order_part = fneq - shear_part

    # Entropic inner products
    inner_sh = jnp.sum(high_order_part * shear_part / (feq + 1e-20), axis=0)
    inner_hh = jnp.sum(high_order_part * high_order_part / (feq + 1e-20), axis=0)

    # Entropic stabilizer γ
    half_gamma = 1.0 / omega - (1.0 - 1.0 / omega) * inner_sh / (inner_hh + 1e-20)

    # KBC collision step
    f -= omega * (shear_part + half_gamma[None, ...] * high_order_part)
    
    return f

