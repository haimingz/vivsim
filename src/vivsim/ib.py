"""This file contains the immersed boundary (IB) method functions for the LBM solver.

Note: Since f is already taken by the distribution functions, we use g for the force field.
"""

import jax.numpy as jnp


# --------------------------------- immersed boundary method ---------------------------------

def stencil2(distance):
    """1D stencil function of range 2 for interpolation."""
    
    return jnp.where(jnp.abs(distance) <= 1, 1 - jnp.abs(distance), 0)


def stencil3(distance):
    """1D stencil function of range 3 for interpolation."""
    
    distance = jnp.abs(distance)
    return jnp.where(
        distance > 1.5,
        0,
        jnp.where(
            distance < 0.5,
            (1 + jnp.sqrt(1 - 3 * distance**2)) / 3,
            (5 - 3 * distance - jnp.sqrt(-2 + 6 * distance - 3 * distance**2)) / 6,
        ),
    )


def kernel2(x, y, X, Y):
    """2D Kernel function of range 2.
    
    Parameters:
    - x, y: The coordinates of a point in fluid.
    - X, Y: The coordinates of the fluid mesh.
    
    Returns:
    - The kernel function of the same size as the fluid mesh.
    """
    
    return stencil2(X - x) * stencil2(Y - y)


def kernel3(x, y, X, Y):
    """2D Kernel function f range 3.
    
    Parameters:
    - x, y: The coordinates of a point in fluid.
    - X, Y: The coordinates of the fluid mesh.
    
    Returns:
    - The kernel function of the same size as the fluid mesh.
    """
    
    return stencil3(X - x) * stencil3(Y - y)


def interpolate_u(u, kernel):
    """Interpolates the fluid velocity at a point in the fluid.
    
    Parameters:
    - u: The velocity field with shape (2, NX, NY).
    - kernel: The kernel function with shape (NX, NY).
    
    Returns:
    - The interpolated velocity at the point.
    """
    
    return jnp.einsum("xy,dxy->d", kernel, u)


def spread_g(g, kernel):
    """Spreads the force to the fluid mesh.
    
    Parameters:
    - g: The force vector with shape (2).
    - kernel: The kernel function with shape (NX, NY).
    
    Returns:
    - The force field with shape (2, NX, NY)."""
    
    return kernel * g[:, None, None]


def get_g_correction(v, u):
    """Computes the correction force to the fluid.    
    g = 2 * rho * (v - u) / dt
    """
    return 2 * (v - u)


def get_u_correction(g):
    """Computes the velocity correction to the fluid.
    du = g * dt / (2 * rho)
    """
    return g * 0.5


def get_source(u, g, omega):
    """Computes the source term needed to be added to the distribution functions.
    
    Parameters:
    - u: The velocity vector with shape (2, NX, Ny).
    - g: The force vector with shape (2, NX, NY).
    - omega: The relaxation parameter.
    
    Returns:
    - The source term with shape (9, NX, NY).
    """

    gxux = g[0] * u[0]
    gyuy = g[1] * u[1]
    gxuy = g[0] * u[1]
    gyux = g[1] * u[0]
    
    foo = (1 - 0.5 * omega)

    _ = jnp.zeros((9, u.shape[1], u.shape[2]))
    
    _ = _.at[0].set(4 / 3 * (-gxux - gyuy)) * foo
    _ = _.at[1].set(1 / 3 * (2 * gxux + g[0] - gyuy)) * foo
    _ = _.at[2].set(1 / 3 * (2 * gyuy + g[1] - gxux)) * foo
    _ = _.at[3].set(1 / 3 * (2 * gxux - g[0] - gyuy)) * foo
    _ = _.at[4].set(1 / 3 * (2 * gyuy - g[1] - gxux)) * foo
    _ = _.at[5].set(1 / 12 * (2 * gxux + 3 * gxuy + g[0] + 3 * gyux + 2 * gyuy + g[1])) * foo
    _ = _.at[6].set(1 / 12 * (2 * gxux - 3 * gxuy - g[0] - 3 * gyux + 2 * gyuy + g[1])) * foo
    _ = _.at[7].set(1 / 12 * (2 * gxux + 3 * gxuy - g[0] + 3 * gyux + 2 * gyuy - g[1])) * foo
    _ = _.at[8].set(1 / 12 * (2 * gxux - 3 * gxuy + g[0] - 3 * gyux + 2 * gyuy - g[1])) * foo

    return _