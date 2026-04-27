"""Guo forcing scheme for the D3Q19 lattice."""

import jax.numpy as jnp

from ..basic import CS2, Q, VELOCITIES, WEIGHTS


def get_guo_forcing_term(g, u):
    """Compute the D3Q19 lattice forcing term using Guo's scheme.

    Args:
        g (jax.Array): External force density with shape ``(3, *spatial_dims)``.
        u (jax.Array): Fluid velocity with shape ``(3, *spatial_dims)``.

    Returns:
        jax.Array: Lattice forcing term with shape ``(19, *spatial_dims)``.
    """

    ndim = g.ndim - 1
    velocity_set = VELOCITIES.reshape((Q, 3) + (1,) * ndim)
    weights = WEIGHTS.reshape((Q,) + (1,) * ndim)

    cu = jnp.einsum("qd,d...->q...", VELOCITIES, u, precision='highest')
    forcing_vector = (
        (velocity_set - u[None, ...]) / CS2
        + velocity_set * (cu[:, None, ...] / (CS2**2))
    )

    return weights * jnp.sum(forcing_vector * g[None, ...], axis=1)


def forcing_guo_bgk(f, g, u, omega):
    """Apply Guo forcing to a BGK / SRT collision step.

    Args:
        f (jax.Array): Distribution function with shape ``(19, *spatial_dims)``.
        g (jax.Array): External force density with shape ``(3, *spatial_dims)``.
        u (jax.Array): Fluid velocity with shape ``(3, *spatial_dims)``.
        omega (float): BGK relaxation parameter.

    Returns:
        jax.Array: Updated distribution function with applied forcing.
    """

    g_lattice = get_guo_forcing_term(g, u)
    return f + g_lattice * (1 - 0.5 * omega)


def get_mrt_forcing_operator(omega):
    """Pre-compute the MRT forcing operator for D3Q19.

    The imported MRT basis is validated against the expected ``(19, 19)``
    D3Q19 shape before constructing the forcing operator.
    """

    from ..collision.mrt import M, M_INV, get_mrt_relaxation_matrix

    q = VELOCITIES.shape[0]
    S = jnp.asarray(get_mrt_relaxation_matrix(omega))
    M = jnp.asarray(M)
    M_INV = jnp.asarray(M_INV)

    if M.shape != (q, q) or M_INV.shape != (q, q) or S.shape != (q, q):
        raise NotImplementedError(
            "Expected lbm3d.collision.mrt to expose D3Q19 operators with "
            f"shape ({q}, {q}); got M={M.shape}, M_INV={M_INV.shape}, S={S.shape}."
        )

    return M_INV @ (jnp.eye(q) - 0.5 * S) @ M


def forcing_guo_mrt(f, g, u, mrt_forcing_operator):
    """Apply Guo forcing to an MRT collision step.

    Args:
        f (jax.Array): Distribution function with shape ``(19, *spatial_dims)``.
        g (jax.Array): External force density with shape ``(3, *spatial_dims)``.
        u (jax.Array): Fluid velocity with shape ``(3, *spatial_dims)``.
        mrt_forcing_operator (jax.Array): MRT forcing operator with shape
            ``(19, 19)``.

    Returns:
        jax.Array: Updated distribution function with applied forcing.
    """

    q = VELOCITIES.shape[0]
    if mrt_forcing_operator.shape != (q, q):
        raise ValueError(
            f"Expected mrt_forcing_operator with shape ({q}, {q}), "
            f"got {mrt_forcing_operator.shape}."
        )

    g_lattice = get_guo_forcing_term(g, u)
    return f + jnp.tensordot(mrt_forcing_operator, g_lattice, axes=([1], [0]), precision='highest')
