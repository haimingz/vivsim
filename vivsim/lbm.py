""" 
This file implements the basic functions for 2-dimensional fluid simulations using
the lattice Boltzmann method (LBM). The implementation includes:

Lattice model:
    * D2Q9 model, with its 9 velocity directions numbered as follows:

    6   2   5
      \ | /
    3 - 0 - 1
      / | \
    7   4   8

Collision Model:
    * Bhatnagar-Gross-Krook (BGK) model, also known as the Single-Relaxation-Time (SRT) model.

Forcing model:
    * Guo forcing scheme for inclusion of external forces.
    
Boundary Conditions (BC):
    * Periodic BC at domain boundaries (automatically enforced by the streaming step).
    * No-slip BC at domain boundaries and obstacles using the Bounce-Back scheme.
    * BC with prescribed velocity at domain boundaries using the Zou/He scheme.
    * No-gradient outlet BC at domain boundaries using the Zou/He scheme.
    * Simple outlet BC at domain boundaries by copying the second last row/column.

Key Variables in this file:
    * rho: Macroscopic density, shape (NX, NY)
    * u: Macroscopic velocity vector, shape (2, NX, NY)
    * g: External force density vector, shape (2, NX, NY)
    * f: Distribution functions, shape (9, NX, NY)
    * feq: Equilibrium distribution functions, shape (9, NX, NY) 
    * g_lattice: forcing term (g discretized into lattice dirs), shape (9, NX, NY)
    where NX and NY are the number of lattice nodes in the x and y directions, respectively.

Note:
    * Some output arrays are created outside the functions and passed in as arguments, 
      so that they can be modified in-place to avoid unnecessary memory allocation/deallocation. 
"""

import jax
import jax.numpy as jnp

def streaming(f):
    """Perform the streaming step by shifting the distribution functions 
    along their respective lattice directions, which automatically enforces
    periodic boundary conditions at domain boundaries.
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
    """Calculate the macroscopic properties (fluid density and velocity)
    based on the distribution functions.
    
    Args:
        f (jax.Array of shape (9, NX, NY)): The distribution functions.
        rho (jax.Array of shape (NX, NY)): The macroscopic density.
        u (jax.Array of shape (2, NX, NY)): The macroscopic velocity.
    
    Returns:
        rho (jax.Array of shape (NX, NY)): The macroscopic density.
        u (jax.Array of shape (2, NX, NY)): The macroscopic velocity.
    """
    
    rho = jnp.sum(f, axis=0)
    u = u.at[0].set((f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho)
    u = u.at[1].set((f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho)
    return rho, u

def get_equilibrium(rho, u, feq):
    """Update the equilibrium distribution function based on the macroscopic properties.
    
    Args:
        rho (jax.Array of shape (NX, NY)): The macroscopic density.
        u (jax.Array of shape (2, NX, NY)): The macroscopic velocity.
        feq (jax.Array of shape (9, NX, NY)): The equilibrium distribution functions.
        
    Returns:
        feq (jax.Array of shape (9, NX, NY)): The equilibrium distribution functions.    
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

def collision(f, feq, omega):
    """Perform the collision step using the single relaxation time (SRT) model. 
    
    Args:
        f (jax.Array of shape (9, NX, NY)): The distribution functions.
        feq (jax.Array of shape (9, NX, NY)): The equilibrium distribution functions.
        omega (scalar): The relaxation parameter (= 1 / relaxation time).
    
    Returns:
        f (jax.Array of shape (9, NX, NY)): The distribution functions after collision.
    """

    return (1 - omega) * f + omega * feq


# --------------------------------- force implementation ---------------------------------

def get_forcing(g, u):
    """Discretize external force density into lattice forcing term
    according to Guo Forcing scheme."""
    
    gxux = g[0] * u[0]
    gyuy = g[1] * u[1]
    gxuy = g[0] * u[1]
    gyux = g[1] * u[0]
    
    forcing = jnp.zeros((9, u.shape[1], u.shape[2]))
    forcing = forcing.at[0].set(4 / 3 * (- gxux - gyuy))
    forcing = forcing.at[1].set(1 / 3 * (2 * gxux + g[0] - gyuy))
    forcing = forcing.at[2].set(1 / 3 * (2 * gyuy + g[1] - gxux))
    forcing = forcing.at[3].set(1 / 3 * (2 * gxux - g[0] - gyuy))
    forcing = forcing.at[4].set(1 / 3 * (2 * gyuy - g[1] - gxux))
    forcing = forcing.at[5].set(1 / 12 * (2 * gxux + 3 * gxuy + g[0] + 3 * gyux + 2 * gyuy + g[1]))
    forcing = forcing.at[6].set(1 / 12 * (2 * gxux - 3 * gxuy - g[0] - 3 * gyux + 2 * gyuy + g[1]))
    forcing = forcing.at[7].set(1 / 12 * (2 * gxux + 3 * gxuy - g[0] + 3 * gyux + 2 * gyuy - g[1]))
    forcing = forcing.at[8].set(1 / 12 * (2 * gxux - 3 * gxuy + g[0] - 3 * gyux + 2 * gyuy - g[1]))
    
    return forcing

def get_velocity_correction(g, rho=1):
    """Compute the velocity correction in the presence of external force. 
    The result should be added to the fluid velocity obtained from `get_macroscopic`
    to preserve second-order accuracy. (Note: The density obtained from `get_macroscopic` 
    does not need to be corrected.)
    
    Args:
        g (jax.Array of shape (2, NX, NY)): The external force density.
        rho (scalar or jax.Array of shape (NX, NY)): The macroscopic density.
        
    Returns:
        out (jax.Array of shape (2, NX, NY)): The velocity correction.
    """

    return g * 0.5 / rho

def get_source(forcing, omega):
    """Compute the source term."""
   
    return forcing * (1 - 0.5 * omega)


# --------------------------------- boundary conditions ---------------------------------

right_dirs = jnp.array([1, 5, 8])
left_dirs = jnp.array([3, 7, 6])
top_dirs = jnp.array([2, 5, 6])
btm_dirs = jnp.array([4, 7, 8])
all_dirs = jnp.array([0,1,2,3,4,5,6,7,8])
opp_dirs = jnp.array([0, 3,4,1,2,7,8,5,6])


# Bounce-back scheme for no-slip boundaries

def left_noslip(f):
    """Enforce a no-slip boundary at the left of the domain 
    using the Bounce Back scheme."""

    return f.at[right_dirs, 0].set(f[left_dirs, 0])

def right_noslip(f):
    """Enforce a no-slip boundary at the right of the domain 
    using the Bounce Back scheme. """

    return f.at[left_dirs, -1].set(f[right_dirs, -1])

def bottom_noslip(f):
    """Enforce a no-slip boundary at the bottom of the domain 
    using the Bounce Back scheme."""

    return f.at[top_dirs, :, 0].set(f[btm_dirs, :, 0])

def top_noslip(f):
    """Enforce a no-slip boundary at the bottom of the domain 
    using the Bounce Back scheme."""

    return f.at[btm_dirs, :, -1].set(f[top_dirs, :, -1])

def obstacle_noslip(f, mask):
    """Enforce a no-slip boundary at the obstacle 
    using the Bounce Back scheme. The obstacle is defined by a 2D mask
    where True indicates the presence of an obstacle."""
    
    return f.at[:, mask].set(f[:, mask][opp_dirs])


# Non-Equilibrium Bounce-Back (or Zou/He) scheme for open boundaries with given velocities

def left_velocity(f, ux, uy):
    """Enforce given velocity ux, uy at the left of the domain
    using the Non-Equilibrium Bounce-Back (or Zou/He) scheme
    where ux, uy can be either scalar or ndarray of shape NY
    """

    rho_wall = (f[0, 0] + f[2, 0] + f[4, 0] + 2 * (f[3, 0] + f[6, 0] + f[7, 0])) / (- ux + 1)
    f = f.at[1, 0].set(f[3, 0] + 2 / 3 * ux * rho_wall)
    f = f.at[5, 0].set(f[7, 0] - 0.5 * (f[2, 0] - f[4, 0]) + (1 / 6 * ux + 0.5 * uy) * rho_wall)
    f = f.at[8, 0].set(f[6, 0] + 0.5 * (f[2, 0] - f[4, 0]) + (1 / 6 * ux - 0.5 * uy) * rho_wall)
    return f

def right_velocity(f, ux, uy):
    """Enforce given velocity ux, uy at the right of the domain
    using the Non-Equilibrium Bounce-Back (or Zou/He) scheme 
    where ux, uy can be either scalar or ndarray of shape NY
    """
    
    rho_wall = (f[0, -1] + f[2, -1] + f[4, -1] + 2 * (f[1, -1] + f[5, -1] + f[8, -1])) / (ux + 1)
    f = f.at[3, -1].set(f[1, -1] - 2 / 3 * ux * rho_wall)
    f = f.at[7, -1].set(f[5, -1] + 0.5 * (f[2, -1] - f[4, -1]) + (- 1 / 6 * ux - 0.5 * uy) * rho_wall)
    f = f.at[6, -1].set(f[8, -1] - 0.5 * (f[2, -1] - f[4, -1]) + (- 1 / 6 * ux + 0.5 * uy) * rho_wall)
    return f

def top_velocity(f, ux, uy):
    """Enforce given velocity ux, uy at the top of the domain
    using the Non-Equilibrium Bounce-Back (or Zou/He) scheme 
    where ux, uy can be either scalar or ndarray of shape NX
    """
    
    rho_wall = (f[0, :,-1] + f[1, :,-1] + f[3, :,-1] + 2 * (f[2, :,-1] + f[5, :,-1] + f[6, :,-1])) / (uy + 1)
    f = f.at[4, :, -1].set(f[2, :, -1] - 2 / 3 * uy * rho_wall)
    f = f.at[7, :, -1].set(f[5, :, -1] + 0.5 * (f[1, :, -1] - f[3, :, -1]) + (1 / 6 * uy - 0.5 * ux) * rho_wall)
    f = f.at[8, :, -1].set(f[6, :, -1] - 0.5 * (f[1, :, -1] - f[3, :, -1]) + (1 / 6 * uy + 0.5 * ux) * rho_wall)
    return f

def bottom_velocity(f, ux, uy):
    """Enforce given velocity ux, uy at the bottom of the domain 
    using the Non-Equilibrium Bounce-Back (or Zou/He) scheme 
    where ux, uy can be either scalar or ndarray of shape NX
    """
    
    rho_wall = (f[0, :,0] + f[1, :,0] + f[3, :,0] + 2 * (f[4, :,0] + f[7, :,0] + f[8, :,0])) / (- uy + 1)
    f = f.at[2, :, 0].set(f[4, :, 0] + 2 / 3 * uy * rho_wall)
    f = f.at[5, :, 0].set(f[7, :, 0] - 0.5 * (f[1, :, 0] - f[3, :, 0]) + (1 / 6 * uy + 0.5 * ux) * rho_wall)
    f = f.at[6, :, 0].set(f[8, :, 0] + 0.5 * (f[1, :, 0] - f[3, :, 0]) + (1 / 6 * uy - 0.5 * ux) * rho_wall)
    return f


# Enforce an outflow boundary by simply copying the second last row/column (1st order accuracy)

def right_outflow(f):
    """Enforce an outflow boundary at the right of the domain
    by just copying the second last row/column."""
    
    return f.at[left_dirs, -1].set(f[left_dirs, -2])

def left_outflow(f):
    """Enforce an outflow boundary at the left of the domain
    by just copying the second last row/column."""

    return f.at[right_dirs, 0].set(f[right_dirs, 1])

def top_outflow(f):
    """Enforce an outflow boundary at the top of the domain
    by just copying the second last row/column."""

    return f.at[top_dirs, :, -1].set(f[top_dirs, :, -2])

def bottom_outflow(f):
    """Enforce an outflow boundary at the bottom of the domain
    by just copying the second last row/column."""

    return f.at[btm_dirs, :, 0].set(f[btm_dirs, :, 1])


# ------------------- cross-device streaming -------------------

def cross_device_stream_y(f, N_DEVICES):
    """Stream fluid particles f from the top/bottom boundary of one device 
    to the bottom/top boundary of the neighbouring devices, assuming the 
    domain is evenly divided among N_DEVICES devices in the y direction."""

    f = f.at[top_dirs, :, 0].set(
        jax.lax.ppermute(f[top_dirs, :, 0], 'y', 
                         [(i, (i + 1) % N_DEVICES) for i in range(N_DEVICES)]))
    f =  f.at[btm_dirs, :, -1].set(
        jax.lax.ppermute(f[btm_dirs, :, -1], 'y', 
                         [((i + 1) % N_DEVICES, i) for i in range(N_DEVICES)]))
    return f 

def cross_device_stream_x(f, N_DEVICES):
    """Stream fluid particles f from the left/right boundary of one device
    to the right/left boundary of the neighbouring devices, assuming the
    domain is evenly divided among N_DEVICES devices in the x direction.
    """

    f = f.at[right_dirs, -1].set(
        jax.lax.ppermute(f[right_dirs, -1], 'y', 
                         [((i + 1) % N_DEVICES, i) for i in range(N_DEVICES)]))
    f = f.at[left_dirs, 0].set(
        jax.lax.ppermute(f[left_dirs, 0], 'y', 
                         [(i, (i + 1) % N_DEVICES) for i in range(N_DEVICES)]))
    return f 