r"""
This file implements the basic operators for 3-dimensional fluid simulations
using the lattice Boltzmann method (LBM).

Lattice model:
    * D3Q19 model with 19 discrete velocity directions.

Collision model:
    * Bhatnagar-Gross-Krook (BGK), also known as the
      Single-Relaxation-Time (SRT) model.

Key variables:
    * rho: macroscopic density, shape (NX, NY, NZ)
    * u: macroscopic velocity vector, shape (3, NX, NY, NZ)
    * f: discrete distribution function (DDF), shape (19, NX, NY, NZ)
    * feq: equilibrium DDF, shape (19, NX, NY, NZ)
    * omega: relaxation parameter, scalar
    * nu: kinematic viscosity in lattice units, scalar

The spatial axes follow the array order (x, y, z).
"""

import jax
import jax.numpy as jnp

from ._lattice import D3Q19

DIM = D3Q19.dim
Q = D3Q19.q
CS2 = D3Q19.cs2

WEIGHTS = jnp.asarray(D3Q19.weights)
VELOCITIES = jnp.asarray(D3Q19.velocities, dtype=jnp.int32)

RIGHT_DIRS = jnp.asarray(D3Q19.right_dirs)
LEFT_DIRS = jnp.asarray(D3Q19.left_dirs)
UP_DIRS = jnp.asarray(D3Q19.up_dirs)
DOWN_DIRS = jnp.asarray(D3Q19.down_dirs)
FRONT_DIRS = jnp.asarray(D3Q19.front_dirs)
BACK_DIRS = jnp.asarray(D3Q19.back_dirs)
ALL_DIRS = jnp.arange(Q)
OPP_DIRS = jnp.asarray(D3Q19.opp_dirs, dtype=jnp.int32)


def streaming(f):
    """Perform the D3Q19 streaming step with periodic wrap-around.

    Args:
        f (jax.Array): Discrete distribution function with shape
            ``(19, NX, NY, NZ)``.

    Returns:
        jax.Array: Streamed distribution function with the same shape as ``f``.
    """

    def shift_fn(f_ch, shift):
        return jnp.roll(f_ch, shift=shift, axis=(0, 1, 2))

    return jax.vmap(shift_fn)(f, VELOCITIES)


def get_macroscopic(f):
    """Recover macroscopic density and velocity from the DDF.

    Args:
        f (jax.Array): Discrete distribution function with shape
            ``(19, *spatial_dims)``.

    Returns:
        tuple[jax.Array, jax.Array]:
            ``rho`` with shape ``(*spatial_dims)`` and ``u`` with shape
            ``(3, *spatial_dims)``.
    """

    rho = jnp.sum(f, axis=0)
    momentum = jnp.einsum("qd,q...->d...", VELOCITIES, f, precision='highest')
    rho_safe = jnp.where(rho == 0, 1, rho)
    u = momentum / rho_safe[None, ...]
    u = jnp.where(rho[None, ...] > 0, u, 0)
    return rho, u


def get_equilibrium(rho, u):
    """Compute the D3Q19 equilibrium distribution.

    Args:
        rho (jax.Array): Macroscopic density with shape ``(*spatial_dims)``.
        u (jax.Array): Macroscopic velocity with shape ``(3, *spatial_dims)``.

    Returns:
        jax.Array: Equilibrium DDF with shape ``(19, *spatial_dims)``.
    """

    ndim = rho.ndim
    uc = jnp.einsum("qd,d...->q...", VELOCITIES, u, precision='highest')
    u_sq = jnp.sum(u**2, axis=0)
    weights = WEIGHTS.reshape((Q,) + (1,) * ndim)

    return rho[None, ...] * weights * (
        1 + uc / CS2 + 0.5 * (uc**2) / (CS2**2) - 0.5 * u_sq[None, ...] / CS2
    )


def collision_bgk(f, feq, omega):
    """Perform the BGK collision step.

    Args:
        f (jax.Array): Discrete distribution function with shape
            ``(19, NX, NY, NZ)``.
        feq (jax.Array): Equilibrium distribution function with the same shape.
        omega (scalar): Relaxation parameter.

    Returns:
        jax.Array: Post-collision distribution function.
    """

    return (1 - omega) * f + omega * feq


def get_omega(nu):
    """Convert kinematic viscosity to the BGK relaxation parameter."""

    return 1 / (3 * nu + 0.5)


def get_velocity_correction(g, rho=1):
    """Return the half-force velocity correction.

    Args:
        g (scalar or jax.Array): External body-force density. Typical full-domain
            shape is ``(3, NX, NY, NZ)``.
        rho (scalar or jax.Array): Macroscopic density.

    Returns:
        scalar or jax.Array: Velocity correction with the same shape as ``g``.
    """

    return g * 0.5 / rho
