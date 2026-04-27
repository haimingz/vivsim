"""Post-processing utilities for fluid fields (2D and 3D)."""

import jax.numpy as jnp


def velocity_magnitude(u):
    """Compute the velocity magnitude field.

    Args:
        u: Velocity field, shape (2, NX, NY) or (3, NX, NY, NZ).
    Returns:
        jax.Array: Scalar velocity magnitude field.
    """
    return jnp.linalg.norm(u, axis=0)


def velocity_gradient(u):
    """Compute the velocity gradient tensor (du_i/dx_j).

    Args:
        u: Velocity field, shape (2, NX, NY) or (3, NX, NY, NZ).
    Returns:
        jax.Array: Gradient tensor of shape (dim, dim, *spatial).
    """
    dim = u.shape[0]
    # Vectorize gradient over all spatial axes
    spatial_axes = tuple(range(1, dim + 1))
    grads = jnp.gradient(u, axis=spatial_axes)
    return jnp.stack(grads, axis=1)


def vorticity(u):
    """Compute the vorticity of the velocity field.

    Returns the scalar field (∂v/∂x - ∂u/∂y) for 2D, or the curl vector
    field for 3D.

    Args:
        u: Velocity field, shape (2, NX, NY) or (3, NX, NY, NZ).
    Returns:
        jax.Array: Scalar field (2D) or vector field (3D).
    """
    if u.shape[0] == 2:
        # 2D scalar vorticity: ∂u/∂y − ∂v/∂x
        return jnp.gradient(u[0], axis=1) - jnp.gradient(u[1], axis=0)

    # 3D: ω = ∇ × u
    dudy = jnp.gradient(u[0], axis=1)
    dudz = jnp.gradient(u[0], axis=2)
    dvdx = jnp.gradient(u[1], axis=0)
    dvdz = jnp.gradient(u[1], axis=2)
    dwdx = jnp.gradient(u[2], axis=0)
    dwdy = jnp.gradient(u[2], axis=1)
    return jnp.stack([dwdy - dvdz, dudz - dwdx, dvdx - dudy])


def vorticity_magnitude(u):
    """Compute the vorticity magnitude field.

    Args:
        u: Velocity field, shape (2, NX, NY) or (3, NX, NY, NZ).
    Returns:
        jax.Array: Scalar field representing absolute vorticity.
    """
    w = vorticity(u)
    if u.shape[0] == 2:
        return jnp.abs(w)
    return jnp.linalg.norm(w, axis=0)



def divergence(u):
    """Compute the divergence of the velocity field.

    Can be used as a diagnostic to check mass conservation (incompressibility).

    Args:
        u: Velocity field, shape (2, NX, NY) or (3, NX, NY, NZ).
    Returns:
        jax.Array: Scalar divergence field.
    """
    dim = u.shape[0]
    return sum(jnp.gradient(u[i], axis=i) for i in range(dim))


def strain_rate(u):
    """Compute the strain-rate (symmetric) tensor.

    Args:
        u: Velocity field, shape (2, NX, NY) or (3, NX, NY, NZ).
    Returns:
        jax.Array: Symmetric strain-rate tensor, shape (dim, dim, *spatial).
    """
    G = velocity_gradient(u)
    return 0.5 * (G + jnp.swapaxes(G, 0, 1))


def strain_rate_magnitude(u):
    """Compute the Frobenius norm of the strain-rate tensor.

    Args:
        u: Velocity field, shape (2, NX, NY) or (3, NX, NY, NZ).
    Returns:
        jax.Array: Scalar strain-rate magnitude field.
    """
    S = strain_rate(u)
    return jnp.sqrt(jnp.einsum('ij...,ij...->...', S, S))


def kinetic_energy(u):
    """Compute the point-wise kinetic energy density field.

    Args:
        u: Velocity field, shape (2, NX, NY) or (3, NX, NY, NZ).
    Returns:
        jax.Array: Scalar kinetic energy field.
    """
    return 0.5 * jnp.sum(u ** 2, axis=0)


def mean_kinetic_energy(u):
    """Compute the domain-averaged kinetic energy density.

    Args:
        u: Velocity field, shape (2, NX, NY) or (3, NX, NY, NZ).
    Returns:
        jax.Array: Scalar value representing the mean kinetic energy.
    """
    return jnp.mean(kinetic_energy(u))


def pressure(rho, cs2=1.0/3.0):
    """Compute pressure from density via the LBM equation of state.

    Args:
        rho: Density field, shape (NX, NY) or (NX, NY, NZ).
        cs2: Speed of sound squared, default is 1/3.
    Returns:
        jax.Array: Scalar pressure field.
    """
    return rho * cs2


def enstrophy(u):
    """Compute the point-wise enstrophy field (vorticity variance).

    Args:
        u: Velocity field, shape (2, NX, NY) or (3, NX, NY, NZ).
    Returns:
        jax.Array: Scalar enstrophy field.
    """
    w = vorticity(u)
    if u.shape[0] == 2:
        return 0.5 * w ** 2
    return 0.5 * jnp.sum(w ** 2, axis=0)


def mean_enstrophy(u):
    """Compute the domain-averaged enstrophy.

    Args:
        u: Velocity field, shape (2, NX, NY) or (3, NX, NY, NZ).
    Returns:
        jax.Array: Scalar value representing the mean enstrophy.
    """
    return jnp.mean(enstrophy(u))


# ====================== Layer 3: Vortex Identification ======================


def q_criterion(u):
    """Compute the Q-criterion for vortex identification.

    Positive Q marks vortex-dominated regions where rotation rate 
    exceeds strain rate.

    Args:
        u: Velocity field, shape (2, NX, NY) or (3, NX, NY, NZ).
    Returns:
        jax.Array: Scalar Q-criterion field.
    """
    G = velocity_gradient(u)
    # The Q-criterion is mathematically equivalent to -0.5 * trace(G^2)
    # which can be elegantly computed avoiding explicit S and O tensors.
    return -0.5 * jnp.einsum('ij...,ji...->...', G, G)



# ====================== Backward Compatibility ======================
# Deprecated aliases — will be removed in a future release.


def calculate_curl(u):
    """Deprecated: use :func:`vorticity` instead."""
    return vorticity(u)


def calculate_vorticity(u):
    """Deprecated: use :func:`vorticity` instead."""
    return vorticity(u)


def calculate_vorticity_dimensionless(u, l, u0):
    """Deprecated: use ``vorticity(u) * l / u0`` instead."""
    return vorticity(u) * l / u0


def calculate_velocity_magnitude(u):
    """Deprecated: use :func:`velocity_magnitude` instead."""
    return velocity_magnitude(u)
