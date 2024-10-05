"""
This file provides the core functions for the lattice Boltzmann method (LBM) in 2D.

Collision models:
* Bhatnagar-Gross-Krook (BGK) model | Single Relaxation Time (SRT) model
* Multiple Relaxation Time (MRT) model

Forcing schemes
* Guo's scheme

Boundary conditions:
* Solid (No-slip) boundary at domain boundaries and obstacles using the bounce-back scheme
* Velocity (Dirichlet) boundary at domain boundaries using the NEBB | Zou/He scheme
* Outflow (No-gradient) boundary at domain boundaries via copying the second last row/column 

All equations have been partially evaluated for the D2Q9 model to maximize efficiency.
"""

import numpy as np
import jax.numpy as jnp

def streaming(f):
    """
    Transport fluid particles to neighboring lattice nodes along their velocity directions.

    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function before streaming.

    Returns:
        f (ndarray of shape (9, NX, NY)): The updated distribution functions after streaming.
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
    """
    Compute the fluid density and velocity according to the distribution functions.

    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.
        rho (ndarray of shape (NX, NY)): The macroscopic density.
        u (ndndarray of shape (2, NX, NY)): The macroscopic velocity.

    Returns:
        rho (ndarray of shape (NX, NY)): The macroscopic density.
        u (ndarray of shape (2, NX, NY)): The macroscopic velocity.
    """
    rho = jnp.sum(f, axis=0)
    u = u.at[0].set((f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho)
    u = u.at[1].set((f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho)
    return rho, u

def get_equilibrium(rho, u, feq):
    """
    Calculate the equilibrium distribution function.

    Args:
        rho (ndarray of shape (NX, NY)): The macroscopic density.
        u (ndarray of shape (2, NX, NY)): The macroscopic velocity.
        feq (ndarray of shape (9, NX, NY)): The equilibrium distribution function.

    Returns:
        feq (ndarray of shape (9, NX, NY)): The updated equilibrium distribution function.
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

# --------------------------------- collision models ---------------------------------

def collision_bgk(f, feq, omega):
    """
    Perform the collision step using the Bhatnagar-Gross-Krook (BGK) model.

    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.
        feq (ndarray of shape (9, NX, NY)): The equilibrium distribution function.
        omega (scalar): The relaxation parameter.

    Returns:
        feq(ndarray of shape (9, NX, NY)): The post-collision distribution function.
    """
    return (1 - omega) * f + omega * feq

def get_omega_mrt(omega):
    """
    Generate the multiple relaxation time (MRT) omega matrix.

    Args:
        omega (scalar): The relaxation parameter.

    Returns:
        omega_mrt (ndarray of shape (9,9)):  The MRT omega matrix.
    """
    # transformation matrix
    M = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [-4, -1, -1, -1, -1, 2, 2, 2, 2],
            [4, -2, -2, -2, -2, 1, 1, 1, 1],
            [0, 1, 0, -1, 0, 1, -1, -1, 1],
            [0, -2, 0, 2, 0, 1, -1, -1, 1],
            [0, 0, 1, 0, -1, 1, 1, -1, -1],
            [0, 0, -2, 0, 2, 1, 1, -1, -1],
            [0, 1, -1, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, 1, -1],
        ]
    )
    # relaxation matrix
    S = np.diag(np.array([1, 1.4, 1.4, 1, 1.2, 1, 1.2, omega, omega]))

    # MRT omega matrix
    return jnp.array(np.linalg.inv(M) @ S @ M)

def collision_mrt(f, feq, omega_mrt):
    """
    Perform collision step using the Multiple Relaxation Time (MRT) model.

    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.
        feq (ndarray of shape (9, NX, NY)): The equilibrium distribution function.
        omega_mrt (ndarray of shape (9, NX, NY)): The relaxation rates for the MRT model.

    Returns:
        feq (ndarray of shape (9, NX, NY)): The updated distribution function after the collision step.
    """
    return jnp.tensordot(omega_mrt, feq - f, axes=([1], [0])) + f

# ----------------- forcing schemes -----------------

def get_u_correction(g, rho=1):
    """Compute the velocity correction according to Guo's scheme.
        du = g * dt / (2 * rho)
    """
    return g * 0.5 / rho


def get_source(u, g, omega):
    """Compute the source term needed to be added to the distribution functions
    according to Guo's scheme.
    
    Args:
        u: The velocity vector with shape (2, NX, NY).
        g: The force vector with shape (2, NX, NY).
        omega: The relaxation parameter.
    
    Returns:
        The source term with shape (9, NX, NY).
    """

    gxux = g[0] * u[0]
    gyuy = g[1] * u[1]
    gxuy = g[0] * u[1]
    gyux = g[1] * u[0]
    
    foo = (1 - 0.5 * omega)

    _ = jnp.zeros((9, u.shape[1], u.shape[2]))
    
    _ = _.at[0].set(4 / 3 * (- gxux - gyuy))
    _ = _.at[1].set(1 / 3 * (2 * gxux + g[0] - gyuy))
    _ = _.at[2].set(1 / 3 * (2 * gyuy + g[1] - gxux))
    _ = _.at[3].set(1 / 3 * (2 * gxux - g[0] - gyuy))
    _ = _.at[4].set(1 / 3 * (2 * gyuy - g[1] - gxux))
    _ = _.at[5].set(1 / 12 * (2 * gxux + 3 * gxuy + g[0] + 3 * gyux + 2 * gyuy + g[1]))
    _ = _.at[6].set(1 / 12 * (2 * gxux - 3 * gxuy - g[0] - 3 * gyux + 2 * gyuy + g[1]))
    _ = _.at[7].set(1 / 12 * (2 * gxux + 3 * gxuy - g[0] + 3 * gyux + 2 * gyuy - g[1]))
    _ = _.at[8].set(1 / 12 * (2 * gxux - 3 * gxuy + g[0] - 3 * gyux + 2 * gyuy - g[1]))

    return _ * foo

# --------------------------------- boundary conditions ---------------------------------
# 6   2   5
#   \ | /
# 3 - 0 - 1
#   / | \
# 7   4   8 

right_indices = jnp.array([1, 5, 8])
left_indices = jnp.array([3, 7, 6])
top_indices = jnp.array([2, 5, 6])
bottom_indices = jnp.array([4, 7, 8])


# Bounce-back scheme for no-slip boundaries

def left_solid(f):
    """Enforce a solid boundary at the left of the domain using the Bounce Back (BB) scheme.

    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.

    Returns:
        f (ndarray of shape (9, NX, NY)): The updated distribution function.
    """

    f = f.at[right_indices, 0].set(f[left_indices, 0])
    return f

def right_solid(f):
    """Enforce a solid boundary at the right of the domain using the Bounce Back (BB) scheme.

    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.

    Returns:
        f (ndarray of shape (9, NX, NY)): The updated distribution function.
    """

    f = f.at[left_indices, -1].set(f[right_indices, -1])
    return f

def bottom_solid(f):
    """Enforce a solid boundary at the bottom of the domain using the Bounce Back (BB) scheme.

    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.

    Returns:
        f (ndarray of shape (9, NX, NY)): The updated distribution function.
    """

    f = f.at[top_indices, :, 0].set(f[bottom_indices, :, 0])
    return f

def top_solid(f):
    """Enforce a solid boundary at the bottom of the domain using the Bounce Back (BB) scheme.

    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.

    Returns:
        f (ndarray of shape (9, NX, NY)): The updated distribution function.
    """

    f = f.at[bottom_indices, :, -1].set(f[top_indices, :, -1])
    return f

def obj_solid(f, mask):
    """Enforce a solid boundary at the object using the Bounce Back (BB) scheme.
    
    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.
        mask (ndarray of shape (NX, NY)): filled with 0 and 1 indicating fluid/solid
    
    Returns:
        f (ndarray of shape (9, NX, NY)): The updated distribution function.
    """
    
    f_ = f
    f_ = f_.at[1, mask].set(f[3, mask])
    f_ = f_.at[2, mask].set(f[4, mask])
    f_ = f_.at[3, mask].set(f[1, mask])
    f_ = f_.at[4, mask].set(f[2, mask])
    f_ = f_.at[5, mask].set(f[7, mask])
    f_ = f_.at[6, mask].set(f[8, mask])
    f_ = f_.at[7, mask].set(f[5, mask])
    f_ = f_.at[8, mask].set(f[6, mask])
    return f_


# Non-Equilibrium Bounce-Back (NEBB, or Zhou/He) scheme for open boundaries with given velocities

def left_velocity(f, ux_left, uy_left):
    """
    Enforce given velocity at the left of the domain 
    using the Non-Equilibrium Bounce-Back (NEBB, or Zhou/He) scheme.

    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.
        ux_left, uy_left (scalar or ndarray of shape NY): The macroscopic velocity at left boundary.

    Returns:
        f (ndarray of shape (9, NX, NY)): The updated distribution function.
    """

    rho_wall = (f[0, 0] + f[2, 0] + f[4, 0] + 2 * (f[3, 0] + f[6, 0] + f[7, 0])) / (- ux_left + 1)
    f = f.at[1, 0].set(f[3, 0] + 2 / 3 * ux_left * rho_wall)
    f = f.at[5, 0].set(f[7, 0] - 0.5 * (f[2, 0] - f[4, 0]) + (1 / 6 * ux_left + 0.5 * uy_left) * rho_wall)
    f = f.at[8, 0].set(f[6, 0] + 0.5 * (f[2, 0] - f[4, 0]) + (1 / 6 * ux_left - 0.5 * uy_left) * rho_wall)
    return f

def right_velocity(f, ux_right, uy_right):
    """
    Enforce given velocity at the right of the domain 
    using the Non-Equilibrium Bounce-Back (NEBB, or Zhou/He) scheme.
    
    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.
        ux_right, uy_right (scalar or ndarray of shape NY): The macroscopic velocity at right boundary.
    
    Returns:
        f (ndarray of shape (9, NX, NY)): The updated distribution function.
    """
    
    rho_wall = (f[0, -1] + f[2, -1] + f[4, -1] + 2 * (f[1, -1] + f[5, -1] + f[8, -1])) / (ux_right + 1)
    f = f.at[3, -1].set(f[1, -1] - 2 / 3 * ux_right * rho_wall)
    f = f.at[7, -1].set(f[5, -1] + 0.5 * (f[2, -1] - f[4, -1]) + (- 1 / 6 * ux_right - 0.5 * uy_right) * rho_wall)
    f = f.at[6, -1].set(f[8, -1] - 0.5 * (f[2, -1] - f[4, -1]) + (- 1 / 6 * ux_right + 0.5 * uy_right) * rho_wall)
    return f

def top_velocity(f, ux_top, uy_top):
    """
    Enforce given velocity at the top of the domain 
    using the Non-Equilibrium Bounce-Back (NEBB, or Zhou/He) scheme.
    
    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.
        ux_top, uy_top (scalar or ndarray of shape NX): The macroscopic velocity at top boundary.
    
    Returns:
        f (ndarray of shape (9, NX, NY)): The updated distribution function.
        rho (ndarray of shape (NX, NY)): The updated macroscopic density.
    """
    
    rho_wall = (f[0, :,-1] + f[1, :,-1] + f[3, :,-1] + 2 * (f[2, :,-1] + f[5, :,-1] + f[6, :,-1])) / (uy_top + 1)
    f = f.at[4, :, -1].set(f[2, :, -1] - 2 / 3 * uy_top * rho_wall)
    f = f.at[7, :, -1].set(f[5, :, -1] + 0.5 * (f[1, :, -1] - f[3, :, -1]) + (1 / 6 * uy_top - 0.5 * ux_top) * rho_wall)
    f = f.at[8, :, -1].set(f[6, :, -1] - 0.5 * (f[1, :, -1] - f[3, :, -1]) + (1 / 6 * uy_top + 0.5 * ux_top) * rho_wall)
    return f

def bottom_velocity(f, ux_bottom, uy_bottom):
    """
    Enforce given velocity at the bottom of the domain 
    using the Non-Equilibrium Bounce-Back (NEBB, or Zhou/He) scheme.
    
    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.
        rho (ndarray of shape (NX, NY)): The macroscopic density.
        ux_bottom, uy_bottom (scalar or ndarray of shape NX): The macroscopic velocity at bottom boundary.
    
    Returns:
        f (ndarray of shape (9, NX, NY)): The updated distribution function.
        rho (ndarray of shape (NX, NY)): The updated macroscopic density.
    """
    
    rho_wall = (f[0, :,0] + f[1, :,0] + f[3, :,0] + 2 * (f[4, :,0] + f[7, :,0] + f[8, :,0])) / (- uy_bottom + 1)
    # rho = rho.at[:, 0].set(rho_wall)
    f = f.at[2, :, 0].set(f[4, :, 0] + 2 / 3 * uy_bottom * rho_wall)
    f = f.at[5, :, 0].set(f[7, :, 0] - 0.5 * (f[1, :, 0] - f[3, :, 0]) + (1 / 6 * uy_bottom + 0.5 * ux_bottom) * rho_wall)
    f = f.at[6, :, 0].set(f[8, :, 0] + 0.5 * (f[1, :, 0] - f[3, :, 0]) + (1 / 6 * uy_bottom - 0.5 * ux_bottom) * rho_wall)
    return f


# Enforce an outflow boundary by simply copying the second last row/column (1st order accuracy)

def right_outflow(f):
    """
    Enforce an outflow boundary at the right of the domain
    by just copying the second last row/column.

    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.

    Returns:
        f (ndarray of shape (9, NX, NY)): The updated distribution function.
    """
    f = f.at[left_indices, -1].set(f[left_indices, -2])
    return f

def left_outflow(f):
    """
    Enforce an outflow boundary at the left of the domain
    by just copying the second last row/column.

    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.

    Returns:
        f (ndarray of shape (9, NX, NY)): The updated distribution function.
    """

    f = f.at[right_indices, 0].set(f[right_indices, 1])
    return f

def top_outflow(f):
    """
    Enforce an outflow boundary at the top of the domain
    by just copying the second last row/column.

    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.

    Returns:
        f (ndarray of shape (9, NX, NY)): The updated distribution function.
    """

    f = f.at[top_indices, :, -1].set(f[top_indices, :, -2])
    return f

def bottom_outflow(f):
    """
    Enforce an outflow boundary at the bottom of the domain
    by just copying the second last row/column.

    Args:
        f (ndarray of shape (9, NX, NY)): The distribution function.

    Returns:
        f (ndarray of shape (9, NX, NY)): The updated distribution function.
    """

    f = f.at[bottom_indices, :, 0].set(f[bottom_indices, :, 1])
    return f
