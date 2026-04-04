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

    fneq_reg = jnp.stack(
        [
            -8.0 * trace_term,
            axis_x,
            axis_y,
            axis_x,
            axis_y,
            diag_pos,
            diag_neg,
            diag_pos,
            diag_neg,
        ],
        axis=0,
    )

    return feq + (1 - omega) * fneq_reg
