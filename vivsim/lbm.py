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


def noslip_boundary(f, loc:str):
    """Enforce a no-slip boundary at the specified boundary using Bounce Back scheme. 
    
    Args:
        f (jax.Array of shape (9, NX, NY)): The distribution functions.
        boundary (str): The boundary where the no-slip condition is enforced, 
            can be 'left', 'right', 'top', or 'bottom'.
        
    Returns:
        f (jax.Array of shape (9, NX, NY)): The distribution functions 
            after enforcing the boundary condition.
    """
    
    if loc == 'left':
        return f.at[RIGHT_DIRS, 0].set(f[LEFT_DIRS, 0])
    if loc == 'right':
        return f.at[LEFT_DIRS, -1].set(f[RIGHT_DIRS, -1])
    if loc == 'top':
        return f.at[UP_DIRS, :, 0].set(f[DOWN_DIRS, :, 0])
    if loc == 'bottom':
        return f.at[UP_DIRS, :, 0].set(f[DOWN_DIRS, :, 0])


def noslip_obstacle(f, mask):
    """Enforce a no-slip boundary at the obstacle 
    using the Bounce Back scheme. The obstacle is defined by a 2D mask
    where True indicates the presence of an obstacle.
    
    Args:
        f (jax.Array of shape (9, NX, NY)): The distribution functions.
        mask (jax.Array of shape (NX, NY)): The mask indicating the obstacle.
    
    Returns:
        f (jax.Array of shape (9, NX, NY)): The distribution functions 
            after enforcing the boundary condition.    
    """
    
    return f.at[:, mask].set(f[:, mask][OPP_DIRS])


def velocity_boundary(f, ux, uy, loc:str):
    """Enforce given velocity ux, uy at the specified boundary using the
    Non-Equilibrium Bounce-Back (or Zou/He) scheme.
    
    Args:
        f (jax.Array of shape (9, NX, NY)): The distribution functions.
        ux (scalar or jax.Array of shape NX): The x-component of velocity.
        uy (scalar or jax.Array of shape NY): The y-component of velocity.
        loc (str): The boundary where the velocity condition is enforced, 
            can be 'left', 'right', 'top', or 'bottom'.
    
    Returns:
        f (jax.Array of shape (9, NX, NY)): The distribution functions 
            after enforcing the boundary condition.
    """
    
    if loc == 'left':        
        rho_wall = (f[0, 0] + f[2, 0] + f[4, 0] + 2 * (f[3, 0] + f[6, 0] + f[7, 0])) / (1 - ux)
    f = f.at[1, 0].set(f[3, 0] + 2 / 3 * ux * rho_wall)
    f = f.at[5, 0].set(f[7, 0] - 0.5 * (f[2, 0] - f[4, 0]) + (1 / 6 * ux + 0.5 * uy) * rho_wall)
    f = f.at[8, 0].set(f[6, 0] + 0.5 * (f[2, 0] - f[4, 0]) + (1 / 6 * ux - 0.5 * uy) * rho_wall)
    return f

    if loc == 'right':        
        rho_wall = (f[0, -1] + f[2, -1] + f[4, -1] + 2 * (f[1, -1] + f[5, -1] + f[8, -1])) / (1 + ux)
    f = f.at[3, -1].set(f[1, -1] - 2 / 3 * ux * rho_wall)
    f = f.at[7, -1].set(f[5, -1] + 0.5 * (f[2, -1] - f[4, -1]) + (- 1 / 6 * ux - 0.5 * uy) * rho_wall)
    f = f.at[6, -1].set(f[8, -1] - 0.5 * (f[2, -1] - f[4, -1]) + (- 1 / 6 * ux + 0.5 * uy) * rho_wall)
    return f

    if loc == 'top':        
        rho_wall = (f[0, :, -1] + f[1, :, -1] + f[3, :, -1] + 2 * (f[2, :, -1] + f[5, :, -1] + f[6, :, -1])) / (1 + uy)
    f = f.at[4, :, -1].set(f[2, :, -1] - 2 / 3 * uy * rho_wall)
    f = f.at[7, :, -1].set(f[5, :, -1] + 0.5 * (f[1, :, -1] - f[3, :, -1]) + (1 / 6 * uy - 0.5 * ux) * rho_wall)
    f = f.at[8, :, -1].set(f[6, :, -1] - 0.5 * (f[1, :, -1] - f[3, :, -1]) + (1 / 6 * uy + 0.5 * ux) * rho_wall)
    return f

    if loc == 'bottom':        
        rho_wall = (f[0, :,0] + f[1, :,0] + f[3, :,0] + 2 * (f[4, :,0] + f[7, :,0] + f[8, :,0])) / (1 - uy)
    f = f.at[2, :, 0].set(f[4, :, 0] + 2 / 3 * uy * rho_wall)
    f = f.at[5, :, 0].set(f[7, :, 0] - 0.5 * (f[1, :, 0] - f[3, :, 0]) + (1 / 6 * uy + 0.5 * ux) * rho_wall)
    f = f.at[6, :, 0].set(f[8, :, 0] + 0.5 * (f[1, :, 0] - f[3, :, 0]) + (1 / 6 * uy - 0.5 * ux) * rho_wall)
    return f


def outlet_boundary(f, loc:str):
    """Enforce a no-gradient BC at the specified boundary using the Zou/He scheme.
    The outflow velocities is computed from the neighboring nodes.
    
    Args:
        f (jax.Array of shape (9, NX, NY)): The distribution functions.
        loc (str): The boundary where the outlet condition is enforced, 
            can be 'left', 'right', 'top', or 'bottom'.
            
    Returns:
        f (jax.Array of shape (9, NX, NY)): The distribution functions 
            after enforcing the boundary condition.
    """

    if loc == 'left':
        rho = jnp.sum(f[:, 0], axis=0)
        ux = (jnp.sum(f[RIGHT_DIRS,0], axis=0) - jnp.sum(f[LEFT_DIRS,0], axis=0)) / rho
        uy = (jnp.sum(f[UP_DIRS,0], axis=0) - jnp.sum(f[DOWN_DIRS,0], axis=0)) / rho
        return velocity_boundary(f, ux, uy, loc)
    
    if loc == 'right':
        rho = jnp.sum(f[:, -1], axis=0)
        ux = (jnp.sum(f[RIGHT_DIRS,-1], axis=0) - jnp.sum(f[LEFT_DIRS,-1], axis=0)) / rho
        uy = (jnp.sum(f[UP_DIRS,-1], axis=0) - jnp.sum(f[DOWN_DIRS,-1], axis=0)) / rho
        return velocity_boundary(f, ux, uy, loc)
    
    if loc == 'top':
        rho = jnp.sum(f[:, :, -1], axis=0)
        ux = (jnp.sum(f[RIGHT_DIRS,:, -1], axis=0) - jnp.sum(f[LEFT_DIRS,:, -1], axis=0)) / rho
        uy = (jnp.sum(f[UP_DIRS,:, -1], axis=0) - jnp.sum(f[DOWN_DIRS,:, -1], axis=0)) / rho
        return velocity_boundary(f, ux, uy, loc)
    
    if loc == 'bottom':
        rho = jnp.sum(f[:, :, 0], axis=0)
        ux = (jnp.sum(f[RIGHT_DIRS,:, 0], axis=0) - jnp.sum(f[LEFT_DIRS,:, 0], axis=0)) / rho
        uy = (jnp.sum(f[UP_DIRS,:, 0], axis=0) - jnp.sum(f[DOWN_DIRS,:, 0], axis=0)) / rho
        return velocity_boundary(f, ux, uy, loc)


def outlet_boundary_simple(f, loc:str):
    """Enforce a no-gradient outlet boundary at the specified boundary 
    by copying the second last row/column.
    
    Args:
        f (jax.Array of shape (9, NX, NY)): The distribution functions.
        loc (str): The boundary where the outlet condition is enforced, 
            can be 'left', 'right', 'top', or 'bottom'.
            
    Returns:
        f (jax.Array of shape (9, NX, NY)): The distribution functions 
            after enforcing the boundary condition.
    """

    if loc == 'left':
        return f.at[LEFT_DIRS, -1].set(f[LEFT_DIRS, -2])
    
    if loc == 'right':
        return f.at[RIGHT_DIRS, 0].set(f[RIGHT_DIRS, 1])

    if loc == 'top':
        return f.at[UP_DIRS, :, -1].set(f[UP_DIRS, :, -2])
    
    if loc == 'bottom':
        return f.at[DOWN_DIRS, :, 0].set(f[DOWN_DIRS, :, 1])
