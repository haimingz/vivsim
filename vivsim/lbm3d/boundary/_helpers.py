r"""Shared helpers and metadata for D3Q19 face boundary conditions."""

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from ..basic import OPP_DIRS, VELOCITIES, get_macroscopic, get_velocity_correction


class BoundarySpec(NamedTuple):
    """Geometry and direction metadata for a single domain face."""

    wall: tuple
    neighbor: tuple
    in_dirs: tuple[int, ...]
    out_dirs: tuple[int, ...]
    zero_dirs: tuple[int, ...]
    reflected_out_dirs: tuple[int, ...]
    tangential_axes: tuple[int, int]
    normal_sign: int
    normal_axis: int


def _make_face_selector(axis: int, index: int):
    selector = [slice(None)] * 3
    selector[axis] = index
    return tuple(selector)


def _find_velocity_index(target):
    matches = np.where(np.all(np.asarray(VELOCITIES) == target, axis=1))[0]
    if len(matches) != 1:
        raise ValueError(f"Failed to identify velocity index for {target}.")
    return int(matches[0])


def _build_boundary_spec(normal_axis: int, normal_sign: int, wall: tuple, neighbor: tuple):
    c = np.asarray(VELOCITIES)
    signed_normal = normal_sign * c[:, normal_axis]

    in_dirs = tuple(np.where(signed_normal > 0)[0].tolist())
    out_dirs = tuple(np.where(signed_normal < 0)[0].tolist())
    zero_dirs = tuple(np.where(signed_normal == 0)[0].tolist())
    tangential_axes = tuple(ax for ax in range(3) if ax != normal_axis)

    reflected_out_dirs = []
    for direction in in_dirs:
        reflected = c[direction].copy()
        reflected[normal_axis] *= -1
        reflected_out_dirs.append(_find_velocity_index(reflected))

    return BoundarySpec(
        wall=wall,
        neighbor=neighbor,
        in_dirs=in_dirs,
        out_dirs=out_dirs,
        zero_dirs=zero_dirs,
        reflected_out_dirs=tuple(reflected_out_dirs),
        tangential_axes=tangential_axes,
        normal_sign=normal_sign,
        normal_axis=normal_axis,
    )


BOUNDARY_SPEC = {
    "left": _build_boundary_spec(
        normal_axis=0,
        normal_sign=1,
        wall=_make_face_selector(0, 0),
        neighbor=_make_face_selector(0, 1),
    ),
    "right": _build_boundary_spec(
        normal_axis=0,
        normal_sign=-1,
        wall=_make_face_selector(0, -1),
        neighbor=_make_face_selector(0, -2),
    ),
    "bottom": _build_boundary_spec(
        normal_axis=1,
        normal_sign=1,
        wall=_make_face_selector(1, 0),
        neighbor=_make_face_selector(1, 1),
    ),
    "top": _build_boundary_spec(
        normal_axis=1,
        normal_sign=-1,
        wall=_make_face_selector(1, -1),
        neighbor=_make_face_selector(1, -2),
    ),
    "back": _build_boundary_spec(
        normal_axis=2,
        normal_sign=1,
        wall=_make_face_selector(2, 0),
        neighbor=_make_face_selector(2, 1),
    ),
    "front": _build_boundary_spec(
        normal_axis=2,
        normal_sign=-1,
        wall=_make_face_selector(2, -1),
        neighbor=_make_face_selector(2, -2),
    ),
}


def get_boundary_shape(f, loc: str):
    """Return the 2D face shape of the selected boundary."""

    spec = BOUNDARY_SPEC[loc]
    return tuple(f.shape[1 + ax] for ax in range(3) if ax != spec.normal_axis)


def get_boundary_size(f, loc: str):
    """Return the number of nodes on the selected boundary face."""

    return int(np.prod(get_boundary_shape(f, loc)))


def _broadcast_to_boundary(value, shape):
    return jnp.broadcast_to(jnp.asarray(value), shape)


def broadcast_wall_values(f, loc: str, rho_wall=1, ux_wall=0, uy_wall=0, uz_wall=0):
    """Broadcast scalar or array wall data to the selected 3D boundary face."""

    shape = get_boundary_shape(f, loc)
    rho_wall = _broadcast_to_boundary(rho_wall, shape)
    ux_wall = _broadcast_to_boundary(ux_wall, shape)
    uy_wall = _broadcast_to_boundary(uy_wall, shape)
    uz_wall = _broadcast_to_boundary(uz_wall, shape)
    return rho_wall, jnp.stack([ux_wall, uy_wall, uz_wall])


def _get_rho_wall_numerator(f, loc: str):
    """Return the common numerator used in face density / velocity recovery."""

    spec = BOUNDARY_SPEC[loc]
    wall = spec.wall
    return jnp.sum(f[jnp.array(spec.zero_dirs), *wall], axis=0) + 2.0 * jnp.sum(
        f[jnp.array(spec.out_dirs), *wall], axis=0
    )


def get_signed_wall_velocity(loc: str, ux_wall=0, uy_wall=0, uz_wall=0):
    """Return inward-normal and tangential wall velocity components."""

    spec = BOUNDARY_SPEC[loc]
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

    spec = BOUNDARY_SPEC[loc]
    rho_wall = _broadcast_to_boundary(rho_wall, get_boundary_shape(f, loc))
    rho_numerator = _get_rho_wall_numerator(f, loc)
    normal_velocity_signed = 1 - rho_numerator / rho_wall

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

    def wrapped(
        f,
        loc: str,
        rho_wall=1,
        ux_wall=0,
        uy_wall=0,
        uz_wall=0,
        gx_wall=0,
        gy_wall=0,
        gz_wall=0,
    ):
        ux_wall, uy_wall, uz_wall = get_corrected_wall_velocity(
            ux_wall,
            uy_wall,
            uz_wall,
            rho_wall=rho_wall,
            gx_wall=gx_wall,
            gy_wall=gy_wall,
            gz_wall=gz_wall,
        )
        return boundary_fn(
            f, loc, rho_wall=rho_wall, ux_wall=ux_wall, uy_wall=uy_wall, uz_wall=uz_wall
        )

    return wrapped
