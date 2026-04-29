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

import jax.numpy as jnp

from .lattice import D3Q19


def shift_x_pos(x):
    return jnp.concatenate((x[:, -1:, :, :], x[:, :-1, :, :]), axis=1)


def shift_x_neg(x):
    return jnp.concatenate((x[:, 1:, :, :], x[:, :1, :, :]), axis=1)


def shift_y_pos(x):
    return jnp.concatenate((x[:, :, -1:, :], x[:, :, :-1, :]), axis=2)


def shift_y_neg(x):
    return jnp.concatenate((x[:, :, 1:, :], x[:, :, :1, :]), axis=2)


def shift_z_pos(x):
    return jnp.concatenate((x[:, :, :, -1:], x[:, :, :, :-1]), axis=3)


def shift_z_neg(x):
    return jnp.concatenate((x[:, :, :, 1:], x[:, :, :, :1]), axis=3)


def streaming(f):
    """Perform the D3Q19 streaming step with periodic wrap-around.

    This implementation spells out the D3Q19 unit shifts with static slices.
    It avoids the generic ``jnp.roll`` path and keeps the work visible to XLA as
    simple concatenate operations.

    Args:
        f (jax.Array): Discrete distribution function with shape
            ``(19, NX, NY, NZ)``.

    Returns:
        jax.Array: Streamed distribution function with the same shape as ``f``.
    """

    return jnp.concatenate((
        f[0:1],
        shift_x_pos(f[1:2]),
        shift_x_neg(f[2:3]),
        shift_y_pos(f[3:4]),
        shift_y_neg(f[4:5]),
        shift_z_pos(f[5:6]),
        shift_z_neg(f[6:7]),
        shift_x_pos(shift_y_pos(f[7:8])),
        shift_x_neg(shift_y_pos(f[8:9])),
        shift_x_pos(shift_y_neg(f[9:10])),
        shift_x_neg(shift_y_neg(f[10:11])),
        shift_x_pos(shift_z_pos(f[11:12])),
        shift_x_neg(shift_z_pos(f[12:13])),
        shift_x_pos(shift_z_neg(f[13:14])),
        shift_x_neg(shift_z_neg(f[14:15])),
        shift_y_pos(shift_z_pos(f[15:16])),
        shift_y_neg(shift_z_pos(f[16:17])),
        shift_y_pos(shift_z_neg(f[17:18])),
        shift_y_neg(shift_z_neg(f[18:19])),
    ), axis=0)


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
    momentum = jnp.einsum("qd,q...->d...", D3Q19.c, f, precision='highest')
    u = momentum / rho[None, ...]
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

    uc = jnp.einsum("qd,d...->q...", D3Q19.c, u, precision="highest")
    weights = D3Q19.w.reshape((D3Q19.q,) + (1,) * ndim)
    feq = rho[None, ...] * weights * (
        1 + uc / D3Q19.cs2
        + 0.5 * (uc**2) / (D3Q19.cs2**2)
        - 0.5 * jnp.sum(u**2, axis=0)[None, ...] / D3Q19.cs2
    )

    return feq


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
