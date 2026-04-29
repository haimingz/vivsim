"""Entropic KBC-style collision operator for the D3Q19 lattice."""

import jax
import jax.numpy as jnp
from .reg import get_second_order_projection


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

    shear_part = get_second_order_projection(fneq)
    high_order_part = fneq - shear_part

    inv_feq = 1.0 / (feq + 1e-20)
    inner_sh = jnp.sum(high_order_part * shear_part * inv_feq, axis=0)
    inner_hh = jnp.sum(high_order_part * high_order_part * inv_feq, axis=0)

    half_gamma = 1.0 / omega - (1.0 - 1.0 / omega) * inner_sh / (inner_hh + 1e-20)

    return f - omega * (
        shear_part + half_gamma[None, ...] * high_order_part
    )
