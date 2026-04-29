"""Bounce-back and specular-reflection face boundaries for the D3Q19 lattice."""

import jax.numpy as jnp

from ..lattice import D3Q19
from ._helpers import broadcast_wall_values


def _moving_wall_correction(in_dirs, rho_wall, u_wall, dtype):
    c_in = D3Q19.c[in_dirs]
    weights = D3Q19.w[in_dirs]
    ndim = u_wall.ndim - 1
    cu_wall = jnp.einsum(
        "id,d...->i...", c_in, u_wall.astype(dtype), precision="highest"
    )
    weights = weights.reshape((len(in_dirs),) + (1,) * ndim)
    return 2.0 * weights * rho_wall[None, ...] * cu_wall / D3Q19.cs2


def boundary_bounce_back(f_before_stream, f, loc, ux_wall=0, uy_wall=0, uz_wall=0):
    """Apply no-slip bounce-back on the selected 3D face."""

    spec = D3Q19.boundary_spec[loc]

    wall_pre = f_before_stream[:, *spec.wall]
    rho_wall = jnp.sum(wall_pre, axis=0).astype(f.dtype)
    _, u_wall = broadcast_wall_values(
        f, loc, rho_wall=1, ux_wall=ux_wall, uy_wall=uy_wall, uz_wall=uz_wall
    )
    correction = _moving_wall_correction(spec.in_dirs, rho_wall, u_wall, f.dtype)

    new_vals = wall_pre[spec.out_dirs] + correction
    new_wall = f[:, *spec.wall].at[spec.in_dirs].set(new_vals)
    return f.at[:, *spec.wall].set(new_wall)


def boundary_specular_reflection(
    f_before_stream, f, loc, ux_wall=0, uy_wall=0, uz_wall=0
):
    """Apply specular reflection on the selected 3D face."""

    spec = D3Q19.boundary_spec[loc]

    wall_pre = f_before_stream[:, *spec.wall]
    rho_wall = jnp.sum(wall_pre, axis=0).astype(f.dtype)
    _, u_wall = broadcast_wall_values(
        f, loc, rho_wall=1, ux_wall=ux_wall, uy_wall=uy_wall, uz_wall=uz_wall
    )
    correction = _moving_wall_correction(spec.in_dirs, rho_wall, u_wall, f.dtype)

    new_vals = wall_pre[spec.out_dirs] + correction
    new_wall = f[:, *spec.wall].at[spec.in_dirs].set(new_vals)
    return f.at[:, *spec.wall].set(new_wall)


def obstacle_bounce_back(f, mask):
    """Apply local bounce-back on obstacle cells selected by a boolean mask."""

    return f.at[:, mask].set(f[:, mask][D3Q19.opp_dirs])
