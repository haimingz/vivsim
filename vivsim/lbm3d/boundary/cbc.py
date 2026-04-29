"""Characteristic face boundary condition for non-reflective D3Q19 boundaries."""

import numpy as np
import jax.numpy as jnp

from ..lattice import D3Q19


def _take_boundary_line(field, axis: int, inward_sign: int):
    """Return the boundary node and the next two fluid nodes along the normal axis."""

    indices = np.array([0, 1, 2] if inward_sign > 0 else [-1, -2, -3], dtype=np.int32)
    return jnp.moveaxis(jnp.take(field, indices, axis=axis), axis, 0)


def boundary_characteristic(rho, u, loc="right"):
    """Apply a one-dimensional characteristic update along the selected face normal."""

    boundary_spec = D3Q19.boundary_spec
    if loc not in boundary_spec:
        raise ValueError(
            "loc must be one of: 'left', 'right', 'bottom', 'top', 'back', 'front'"
        )

    spec = boundary_spec[loc]
    normal_axis = spec.normal_axis
    tangential_axes = spec.tangential_axes
    inward_sign = spec.normal_sign
    cs = jnp.sqrt(D3Q19.cs2)

    rho_b1, rho_b2, rho_b3 = _take_boundary_line(rho, normal_axis, inward_sign)
    un_b1, un_b2, un_b3 = _take_boundary_line(u[normal_axis], normal_axis, inward_sign)
    ut0_b1 = _take_boundary_line(u[tangential_axes[0]], normal_axis, inward_sign)[0]
    ut1_b1 = _take_boundary_line(u[tangential_axes[1]], normal_axis, inward_sign)[0]

    deriv_coeff = -0.5 * inward_sign
    drho_dn = deriv_coeff * (3 * rho_b1 - 4 * rho_b2 + rho_b3)
    dun_dn = deriv_coeff * (3 * un_b1 - 4 * un_b2 + un_b3)

    characteristic_out = dun_dn - inward_sign * cs / rho_b1 * drho_dn
    l_out = (un_b1 - inward_sign * cs) * characteristic_out

    rho_next = rho_b1 - 0.5 * rho_b1 / cs * l_out
    un_next = un_b1 - 0.5 * l_out

    velocity_next = [None, None, None]
    velocity_next[normal_axis] = un_next
    velocity_next[tangential_axes[0]] = ut0_b1
    velocity_next[tangential_axes[1]] = ut1_b1

    return rho_next, jnp.stack(velocity_next)
