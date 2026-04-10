"""Regularized BGK collision operator for the D3Q19 lattice."""

import jax.numpy as jnp

from ..basic import CS2, VELOCITIES, WEIGHTS


def collision_reg(f, feq, omega):
    """Apply second-order regularized BGK collision for D3Q19.

    The non-equilibrium part is projected onto the second-order Hermite
    subspace through the non-equilibrium momentum-flux tensor. This removes
    higher-order ghost content before the BGK relaxation step.

    Args:
        f (jax.Array): Distribution function with shape ``(19, *spatial_dims)``.
        feq (jax.Array): Equilibrium distribution with the same shape as ``f``.
        omega (float): Relaxation parameter.

    Returns:
        jax.Array: Post-collision distribution with the same shape as ``f``.
    """

    fneq = f - feq

    c = VELOCITIES.astype(f.dtype)
    weights = WEIGHTS.astype(f.dtype)
    dim = c.shape[1]
    ndim = f.ndim - 1

    # Second-order non-equilibrium moment tensor:
    # Pi_neq[a, b] = sum_i fneq_i * c_i[a] * c_i[b]
    pi_neq = jnp.einsum("q...,qa,qb->ab...", fneq, c, c)

    identity = jnp.eye(dim, dtype=f.dtype)
    q_tensor = (
        c[:, :, None] * c[:, None, :] - CS2 * identity[None, :, :]
    ).reshape((c.shape[0], dim, dim) + (1,) * ndim)

    # Second-order Hermite projection:
    # fneq_reg_i = w_i / (2 * cs^4) * Q_i : Pi_neq
    coeff = weights.reshape((weights.shape[0],) + (1,) * ndim) / (2 * CS2**2)
    fneq_reg = coeff * jnp.sum(q_tensor * pi_neq[None, ...], axis=(1, 2))

    return feq + (1 - omega) * fneq_reg
