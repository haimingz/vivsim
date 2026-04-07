"""Characteristic boundary condition (CBC) for non-reflective D2Q9 boundaries."""

import jax.numpy as jnp

from ._helpers import BOUNDARY_SPEC


def _take_boundary_line(field, axis: int, inward_sign: int):
    """Return the boundary node and the next two fluid nodes along the normal axis."""
    indices = jnp.array([0, 1, 2]) if inward_sign > 0 else jnp.array([-1, -2, -3])
    return jnp.moveaxis(jnp.take(field, indices, axis=axis), axis, 0)


def boundary_characteristic(rho, u, loc="right"):
    """Non-reflective characteristic boundary condition in lattice units.

    Args:
        rho: Density field with shape ``(NX, NY)``.
        u: Velocity field with shape ``(2, NX, NY)``.
        loc: Boundary location, one of ``{"left", "right", "top", "bottom"}``.

    Returns:
        Tuple ``(rho_next, u_next)`` where ``rho_next`` has shape ``(N,)`` and
        ``u_next`` has shape ``(2, N)`` along the selected boundary.
    """
    if loc not in BOUNDARY_SPEC:
        raise ValueError("loc must be one of: 'left', 'right', 'top', 'bottom'")

    spec = BOUNDARY_SPEC[loc]
    normal_axis = spec.normal_axis
    tangential_axis = 1 - normal_axis
    inward_sign = spec.normal_sign
    cs = 1 / jnp.sqrt(3)

    rho_b1, rho_b2, rho_b3 = _take_boundary_line(rho, normal_axis, inward_sign)
    un_b1, un_b2, un_b3 = _take_boundary_line(u[normal_axis], normal_axis, inward_sign)
    ut_b1 = jnp.take(u[tangential_axis], 0 if inward_sign > 0 else -1, axis=normal_axis)

    deriv_coeff = -0.5 * inward_sign
    drho_dn = deriv_coeff * (3 * rho_b1 - 4 * rho_b2 + rho_b3)
    dun_dn = deriv_coeff * (3 * un_b1 - 4 * un_b2 + un_b3)

    characteristic_out = dun_dn - inward_sign * cs / rho_b1 * drho_dn
    l_out = (un_b1 - inward_sign * cs) * characteristic_out

    rho_next = rho_b1 - 0.5 * rho_b1 / cs * l_out
    un_next = un_b1 - 0.5 * l_out

    velocity_next = [None, None]
    velocity_next[normal_axis] = un_next
    velocity_next[tangential_axis] = ut_b1

    return rho_next, jnp.stack(velocity_next)
