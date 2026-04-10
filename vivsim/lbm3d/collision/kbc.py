"""Entropic KBC-style collision operator for the D3Q19 lattice."""

import jax
import jax.numpy as jnp

from ..basic import CS2, VELOCITIES, WEIGHTS


def _get_second_order_projection(fneq):
    """Project the non-equilibrium DDF onto its second-order Hermite subspace."""

    c = VELOCITIES.astype(fneq.dtype)
    weights = WEIGHTS.astype(fneq.dtype)
    dim = c.shape[1]
    ndim = fneq.ndim - 1

    pi_neq = jnp.einsum("q...,qa,qb->ab...", fneq, c, c, precision='highest')
    identity = jnp.eye(dim, dtype=fneq.dtype)
    q_tensor = (
        c[:, :, None] * c[:, None, :] - CS2 * identity[None, :, :]
    ).reshape((c.shape[0], dim, dim) + (1,) * ndim)

    coeff = weights.reshape((weights.shape[0],) + (1,) * ndim) / (2 * CS2**2)
    return coeff * jnp.sum(q_tensor * pi_neq[None, ...], axis=(1, 2))


def collision_kbc(f: jax.Array, feq: jax.Array, omega: float) -> jax.Array:
    """Apply a D3Q19 KBC-style entropic collision step.

    This implementation splits the non-equilibrium distribution into:
    - a hydrodynamic second-order part, obtained from the non-equilibrium
      momentum-flux tensor, and
    - a remaining higher-order part carrying ghost modes.

    The higher-order part is then relaxed with the adaptive entropic parameter
    ``gamma`` following the usual KBC construction.

    Args:
        f (jax.Array): Current distribution function with shape
            ``(19, *spatial_dims)``.
        feq (jax.Array): Equilibrium distribution with the same shape as ``f``.
        omega (float): Relaxation parameter ``1 / tau``.

    Returns:
        jax.Array: Post-collision distribution function.
    """

    fneq = f - feq

    shear_part = _get_second_order_projection(fneq)
    high_order_part = fneq - shear_part

    inv_feq = 1.0 / (feq + 1e-20)
    inner_sh = jnp.sum(high_order_part * shear_part * inv_feq, axis=0)
    inner_hh = jnp.sum(high_order_part * high_order_part * inv_feq, axis=0)

    half_gamma = 1.0 / omega - (1.0 - 1.0 / omega) * inner_sh / (inner_hh + 1e-20)

    return f - omega * (
        shear_part + half_gamma[None, ...] * high_order_part
    )
