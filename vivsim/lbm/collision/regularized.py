import jax.numpy as jnp


def collision_regularized(f, feq, omega):
    """
    Regularized BGK collision operator for the D2Q9 lattice.

    Projects the non-equilibrium distribution onto its second-order Hermite
    subspace using the closed-form D2Q9 moments.

    Args:
        f (jax.Array): Distribution function with shape (9, NX, NY).
        feq (jax.Array): Equilibrium distribution with shape (9, NX, NY).
        omega (float): Relaxation parameter.

    Returns:
        jax.Array: Post-collision distribution with shape (9, NX, NY).
    """

    fneq = f - feq

    # Second-order non-equilibrium moments for D2Q9.
    diag_sum = fneq[5] + fneq[6] + fneq[7] + fneq[8]
    pi_xx = fneq[1] + fneq[3] + diag_sum
    pi_yy = fneq[2] + fneq[4] + diag_sum
    pi_xy = fneq[5] - fneq[6] + fneq[7] - fneq[8]

    trace_term = (pi_xx + pi_yy) / 12.0
    axis_x = (2.0 * pi_xx - pi_yy) / 6.0
    axis_y = (2.0 * pi_yy - pi_xx) / 6.0
    diag_pos = trace_term + pi_xy / 4.0
    diag_neg = trace_term - pi_xy / 4.0

    fneq_reg = jnp.zeros_like(f)
    fneq_reg = fneq_reg.at[0].set(-8.0 * trace_term)
    fneq_reg = fneq_reg.at[1].set(axis_x)
    fneq_reg = fneq_reg.at[2].set(axis_y)
    fneq_reg = fneq_reg.at[3].set(axis_x)
    fneq_reg = fneq_reg.at[4].set(axis_y)
    fneq_reg = fneq_reg.at[5].set(diag_pos)
    fneq_reg = fneq_reg.at[6].set(diag_neg)
    fneq_reg = fneq_reg.at[7].set(diag_pos)
    fneq_reg = fneq_reg.at[8].set(diag_neg)

    return feq + (1 - omega) * fneq_reg
