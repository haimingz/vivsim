r"""Shared helpers and metadata for 3D face boundary conditions."""

import jax.numpy as jnp

from ..lattice import D3Q19
from ..basic import get_macroscopic, get_velocity_correction


def get_boundary_shape(f, loc: str):
    """Return the 2D face shape of the selected boundary."""

    spec = D3Q19.boundary_spec[loc]
    return tuple(
        f.shape[1 + ax] for ax in range(D3Q19.d) if ax != spec.normal_axis
    )


def broadcast_wall_values(f, loc: str, rho_wall=1, ux_wall=0, uy_wall=0, uz_wall=0):
    """Broadcast scalar or array wall data to the selected 3D boundary face."""

    shape = get_boundary_shape(f, loc)

    if jnp.isscalar(rho_wall):
        rho_wall = jnp.full(shape, rho_wall)
    if jnp.isscalar(ux_wall):
        ux_wall = jnp.full(shape, ux_wall)
    if jnp.isscalar(uy_wall):
        uy_wall = jnp.full(shape, uy_wall)
    if jnp.isscalar(uz_wall):
        uz_wall = jnp.full(shape, uz_wall)

    return rho_wall, jnp.stack([ux_wall, uy_wall, uz_wall])


def _get_rho_wall_numerator(f, loc: str):
    """Return the common numerator used in face density / velocity recovery."""

    spec = D3Q19.boundary_spec[loc]
    wall = spec.wall
    return jnp.sum(f[spec.zero_dirs, *wall], axis=0) + 2.0 * jnp.sum(
        f[spec.out_dirs, *wall], axis=0
    )


def get_signed_wall_velocity(loc: str, ux_wall=0, uy_wall=0, uz_wall=0):
    """Return inward-normal and tangential wall velocity components."""

    spec = D3Q19.boundary_spec[loc]
    velocity = [jnp.asarray(ux_wall), jnp.asarray(uy_wall), jnp.asarray(uz_wall)]
    normal_velocity = spec.normal_sign * velocity[spec.normal_axis]
    tangential_velocity = jnp.stack(
        [spec.normal_sign * velocity[ax] for ax in spec.tangential_axes]
    )
    return normal_velocity, tangential_velocity


def get_rho_wall_from_velocity(f, loc: str, ux_wall=0, uy_wall=0, uz_wall=0):
    """Recover wall density from a prescribed face velocity."""

    normal_velocity, _ = get_signed_wall_velocity(
        loc, ux_wall=ux_wall, uy_wall=uy_wall, uz_wall=uz_wall
    )
    return _get_rho_wall_numerator(f, loc) / (1 - normal_velocity)


def get_wall_velocity_from_pressure(f, loc: str, rho_wall=1):
    """Recover wall velocity from a prescribed face density."""

    spec = D3Q19.boundary_spec[loc]
    rho_wall = jnp.full(get_boundary_shape(f, loc), rho_wall)
    normal_velocity_signed = 1 - _get_rho_wall_numerator(f, loc) / rho_wall

    _, u_neighbor = get_macroscopic(f[:, *spec.neighbor])
    velocity = [u_neighbor[0], u_neighbor[1], u_neighbor[2]]
    velocity[spec.normal_axis] = spec.normal_sign * normal_velocity_signed

    return tuple(velocity)


def get_corrected_wall_velocity(
    ux_wall, uy_wall, uz_wall, rho_wall=1, gx_wall=0, gy_wall=0, gz_wall=0
):
    """Apply the half-force wall-velocity correction component-wise."""

    ux_corrected = ux_wall - get_velocity_correction(gx_wall, rho_wall)
    uy_corrected = uy_wall - get_velocity_correction(gy_wall, rho_wall)
    uz_corrected = uz_wall - get_velocity_correction(gz_wall, rho_wall)
    return ux_corrected, uy_corrected, uz_corrected


def wrap_velocity(boundary_fn):
    """Build a velocity-boundary wrapper around a core boundary kernel."""

    def wrapped(f, loc: str, ux_wall=0, uy_wall=0, uz_wall=0):
        rho_wall = get_rho_wall_from_velocity(
            f, loc, ux_wall=ux_wall, uy_wall=uy_wall, uz_wall=uz_wall
        )
        return boundary_fn(
            f, loc, rho_wall=rho_wall, ux_wall=ux_wall, uy_wall=uy_wall, uz_wall=uz_wall
        )

    return wrapped


def wrap_pressure(boundary_fn):
    """Build a pressure-boundary wrapper around a core boundary kernel."""

    def wrapped(f, loc: str, rho_wall=1):
        ux_wall, uy_wall, uz_wall = get_wall_velocity_from_pressure(
            f, loc, rho_wall=rho_wall
        )
        return boundary_fn(
            f, loc, rho_wall=rho_wall, ux_wall=ux_wall, uy_wall=uy_wall, uz_wall=uz_wall
        )

    return wrapped


def wrap_force_corrected(boundary_fn):
    """Build a force-corrected boundary wrapper around a core kernel."""

    def wrapped(f, loc: str, rho_wall=1, ux_wall=0, uy_wall=0, uz_wall=0,
        gx_wall=0, gy_wall=0, gz_wall=0,
    ):
        ux_wall, uy_wall, uz_wall = get_corrected_wall_velocity(
            ux_wall, uy_wall, uz_wall,
            rho_wall=rho_wall, gx_wall=gx_wall, gy_wall=gy_wall, gz_wall=gz_wall,
        )
        return boundary_fn(
            f, loc, rho_wall=rho_wall, ux_wall=ux_wall, uy_wall=uy_wall, uz_wall=uz_wall
        )

    return wrapped
