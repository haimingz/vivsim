"""This files provides the core functions for the lattice Boltzmann method (LBM) in 2D.

All functions are pure functions as required by JAX. 
All equations have been partially evaluated for the D2Q9 model to maximize efficiency.
"""

import jax 
import jax.numpy as jnp
import numpy as np

# --------------------------------- core LBM functions ---------------------------------

def equilibrum(rho, u, feq):
    """Calculates the equilibrium distribution function.

    Parameters:
    - rho: The density with shape (NX, NY).
    - u: The velocity vector with shape (2, NX, NY).
    - feq: The equilibrium distribution function with shape (9, NX, NY).

    Returns:
    - feq: The updated equilibrium distribution function with shape (9, NX, NY).
    """
    
    uxx = u[0] * u[0]
    uyy = u[1] * u[1]
    uxy = u[0] * u[1]
    uu = uxx + uyy
    feq = feq.at[0].set(4 / 9 * rho * (1 - 1.5 * uu))
    feq = feq.at[1].set(1 / 9 * rho * (1 - 1.5 * uu + 3 * u[0] + 4.5 * uxx))
    feq = feq.at[2].set(1 / 9 * rho * (1 - 1.5 * uu + 3 * u[1] + 4.5 * uyy))
    feq = feq.at[3].set(1 / 9 * rho * (1 - 1.5 * uu - 3 * u[0] + 4.5 * uxx))
    feq = feq.at[4].set(1 / 9 * rho * (1 - 1.5 * uu - 3 * u[1] + 4.5 * uyy))
    feq = feq.at[5].set(1 / 36 * rho * (1 + 3 * uu + 3 * (u[0] + u[1]) + 9 * uxy))
    feq = feq.at[6].set(1 / 36 * rho * (1 + 3 * uu - 3 * (u[0] - u[1]) - 9 * uxy))
    feq = feq.at[7].set(1 / 36 * rho * (1 + 3 * uu - 3 * (u[0] + u[1]) + 9 * uxy))
    feq = feq.at[8].set(1 / 36 * rho * (1 + 3 * uu + 3 * (u[0] - u[1]) - 9 * uxy))    
    return feq


def streaming(f):
    """Performs the streaming step.

    Parameters:
    - f: The distribution function with shape (9, NX, NY).

    Returns:
    - f: The updated distribution functions after streaming.
    """
    
    f = f.at[1].set(jnp.roll(f[1],  1, axis=0))
    f = f.at[2].set(jnp.roll(f[2],  1, axis=1))
    f = f.at[3].set(jnp.roll(f[3], -1, axis=0))
    f = f.at[4].set(jnp.roll(f[4], -1, axis=1))
    f = f.at[5].set(jnp.roll(f[5],  1, axis=0))
    f = f.at[5].set(jnp.roll(f[5],  1, axis=1))
    f = f.at[6].set(jnp.roll(f[6], -1, axis=0))
    f = f.at[6].set(jnp.roll(f[6],  1, axis=1))
    f = f.at[7].set(jnp.roll(f[7], -1, axis=0))
    f = f.at[7].set(jnp.roll(f[7], -1, axis=1))
    f = f.at[8].set(jnp.roll(f[8],  1, axis=0))
    f = f.at[8].set(jnp.roll(f[8], -1, axis=1))
    return f


def get_macroscopic(f, rho, u):
    """Computes the macroscopic density and velocity from the distribution functions.

    Parameters:
    - f: The distribution functions.
    - rho: The density.
    - u: The velocity vector.

    Returns:
    - rho: The density.
    - u: The velocity vector.
    """
    rho = jnp.sum(f, axis=0)
    u = u.at[0].set((f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho)
    u = u.at[1].set((f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho)
    return rho, u


def collision_bgk(f, feq, omega):
    """Performs the BGK collision step.

    Parameters:
    - f: The distribution function with shape (9, NX, NY).
    - feq: The equilibrium distribution function with shape (9, NX, NY).
    - omega (float): The relaxation parameter.

    Returns:
    - The post-collision distribution function with shape (9, NX, NY).
    """
    return (1 - omega) * f + omega * feq


def get_omega_mrt(omega):
    """Generates the MRT collision matrix for a given value of omega.

    Parameters:
    - omega: A scalar value representing the relaxation parameter.

    Returns:
    - The MRT collision matrix.
    """
    # transformation matrix
    M = np.array(
        [
            [ 1,  1,  1,  1,  1,  1,  1,  1,  1],
            [-4, -1, -1, -1, -1,  2,  2,  2,  2],
            [ 4, -2, -2, -2, -2,  1,  1,  1,  1],
            [ 0,  1,  0, -1,  0,  1, -1, -1,  1],
            [ 0, -2,  0,  2,  0,  1, -1, -1,  1],
            [ 0,  0,  1,  0, -1,  1,  1, -1, -1],
            [ 0,  0, -2,  0,  2,  1,  1, -1, -1],
            [ 0,  1, -1,  1, -1,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  1, -1,  1, -1],
        ]
    )
    # relaxation matrix
    S = np.diag(np.array([1, 1.4, 1.4, 1, 1.2, 1, 1.2, omega, omega]))

    # MRT collision matrix
    return jnp.array(np.linalg.inv(M) @ S @ M)


def collision_mrt(f, feq, omega_mrt):
    """Performs collision step using the Multiple Relaxation Time (MRT) model.

    Args:
        f (ndarray): The distribution function with shape (9, NX, NY).
        feq (ndarray): The equilibrium distribution function with shape (9, NX, NY).
        omega_mrt (ndarray): The relaxation rates for the MRT model with shape (9,9).

    Returns:
        ndarray: The updated distribution function after the collision step.
    """
    return jnp.tensordot(omega_mrt, feq - f, axes=([1], [0])) + f


# --------------------------------- boundary conditions ---------------------------------

def left_inlet(f, rho, u, u_inlet):
    """Applies inlet BC at the left of the domain based on Zou/He model.

    Parameters:
    - f (ndarray): Distribution functions representing the particle populations with shape (9, NX, NY).
    - rho (ndarray): Density of the fluid with shape (NX, NY).
    - u (ndarray): Velocity of the fluid with shape (2, NX, NY).
    - u_inlet (float): Inlet velocity.

    Returns:
    - f (ndarray): Updated distribution functions after applying the boundary conditions.
    - rho (ndarray): Updated density after applying the boundary conditions.
    - u (ndarray): Updated velocity after applying the boundary conditions.
    """

    rho_profile = (f[0, 0] + f[2, 0] + f[4, 0] + 2 * (f[3, 0] + f[6, 0] + f[7, 0])) / (
        1 - u_inlet
    )
    rho = rho.at[0].set(rho_profile)
    u = u.at[0, 0].set(u_inlet)
    u = u.at[1, 0].set(0)
    f = f.at[1, 0].set(f[3, 0] + 2 / 3 * rho_profile * u_inlet)
    f = f.at[5, 0].set(f[7, 0] - 0.5 * (f[2, 0] - f[4, 0]) + 1 / 6 * rho_profile * u_inlet)
    f = f.at[8, 0].set(f[6, 0] + 0.5 * (f[2, 0] - f[4, 0]) + 1 / 6 * rho_profile * u_inlet)
    return f, rho, u


def right_outlet(f):
    """Applies outlet BC at the right of the domain (No gradient BC).

    Parameters:
    - f: Distribution functions with shape (9, NX, NY).

    Returns:
    - f: Updated distribution functions after applying the boundary conditions.
    """

    f = f.at[3, -1].set(f[3, -2])
    f = f.at[6, -1].set(f[6, -2])
    f = f.at[7, -1].set(f[7, -2])
    return f


# --------------------------------- immersed boundary method ---------------------------------

def stencil2(distance):
    """Simple stencil function of range 2 for interpolation."""
    
    return jnp.where(jnp.abs(distance) <= 1, 1 - jnp.abs(distance), 0)


def stencil3(distance):
    """Simple stencil function of range 3 for interpolation."""
    
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
    """Computes the correction to the distribution functions.
    
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