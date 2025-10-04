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
    * f: Discrete Distribution Function (DDF), shape (9, NX, NY)
    * feq: Equilibrium DDF, shape (9, NX, NY) 
    * g: External force vector, shape (2, NX, NY)
    * g_lattice: External force discretized into lattice dirs, shape (9, NX, NY)
    where NX and NY are the numbers of lattice nodes in the x and y directions, respectively.

"""

import jax.numpy as jnp

WEIGHTS = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

VELOCITIES = jnp.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
    [1, 1],
    [-1, 1],
    [-1, -1],
    [1, -1]
])

RIGHT_DIRS = jnp.array([1, 5, 8])
LEFT_DIRS = jnp.array([3, 7, 6])
UP_DIRS = jnp.array([2, 5, 6])
DOWN_DIRS = jnp.array([4, 7, 8])
ALL_DIRS = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
OPP_DIRS = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def streaming(f):
    """Perform the streaming step by shifting the DDF 
    along their respective velocity directions, which automatically enforces
    periodic boundary conditions at domain boundaries.
    
    Args:
        f (jax.Array): Discrete distribution function (DDF)
        
    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF after streaming
    """
    
    f = f.at[RIGHT_DIRS].set(jnp.roll(f[RIGHT_DIRS], 1, axis=1))
    f = f.at[LEFT_DIRS].set(jnp.roll(f[LEFT_DIRS], -1, axis=1))
    f = f.at[UP_DIRS].set(jnp.roll(f[UP_DIRS], 1, axis=2))
    f = f.at[DOWN_DIRS].set(jnp.roll(f[DOWN_DIRS], -1, axis=2))
    return f


def get_macroscopic(f):
    """Calculate the macroscopic properties (fluid density and velocity).
    
    Args:
        f (jax.Array): Discrete distribution function (DDF)
        rho (jax.Array of shape (NX, NY)): The macroscopic density.
        u (jax.Array of shape (2, NX, NY)): The macroscopic velocity.
    
    Returns:
        rho (jax.Array of shape (NX, NY)): The macroscopic density.
        u (jax.Array of shape (2, NX, NY)): The macroscopic velocity.
    """
    
    rho = jnp.sum(f, axis=0)
    u = jnp.zeros((2, *rho.shape))
    u = u.at[0].set((jnp.sum(f[RIGHT_DIRS], axis=0) - jnp.sum(f[LEFT_DIRS], axis=0)) / rho)
    u = u.at[1].set((jnp.sum(f[UP_DIRS], axis=0) - jnp.sum(f[DOWN_DIRS], axis=0)) / rho)
    return rho, u


def get_equilibrium(rho, u):
    """Update the equilibrium distribution function based on the macroscopic properties.
    
    Args:
        rho (jax.Array of shape (NX, NY)): The macroscopic density.
        u (jax.Array of shape (2, NX, NY)): The macroscopic velocity.
        feq (jax.Array of shape (9, NX, NY)): The equilibrium DDF.
        
    Returns:
        feq (jax.Array of shape (9, NX, NY)): The equilibrium DDF.    
    """

    uc = (u[0, :, :] * VELOCITIES[:, 0, jnp.newaxis, jnp.newaxis] + 
          u[1, :, :] * VELOCITIES[:, 1, jnp.newaxis, jnp.newaxis])
    
    feq = rho[jnp.newaxis, ...] * WEIGHTS[:, jnp.newaxis, jnp.newaxis] * (
            1 + 3 * uc + 4.5 * uc ** 2 
            - 1.5 * (u[jnp.newaxis, 0] ** 2 + u[jnp.newaxis, 1] ** 2)
        )
    return feq


def collision(f, feq, omega):
    """Perform the collision step using the single relaxation time (SRT) model. 
    
    Args:
        f (jax.Array): Discrete distribution function (DDF)
        feq (jax.Array of shape (9, NX, NY)): The equilibrium DDF.
        omega (scalar): The relaxation parameter (= 1 / relaxation time).
    
    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF after collision.
    """

    return (1 - omega) * f + omega * feq


# --------------------------------- force implementation ---------------------------------


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


def get_discretized_force(g, u):
    """Discretize external force density into lattice forcing term using Guo Forcing scheme.
    
    Args:
        g (jax.Array of shape (2, NX, NY)): The external force density.
        u (jax.Array of shape (2, NX, NY)): The fluid velocity.
    
    Returns:
        g_lattice (jax.Array of shape (9, NX, NY)): The discretize external force term.    
    """
    
    uc = (u[0, :, :] * VELOCITIES[:, 0, jnp.newaxis, jnp.newaxis] + 
          u[1, :, :] * VELOCITIES[:, 1, jnp.newaxis, jnp.newaxis])
    
    g_lattice = WEIGHTS[..., jnp.newaxis, jnp.newaxis] * (
        g[0] * (
            3 * (VELOCITIES[:, 0, jnp.newaxis, jnp.newaxis] - u[jnp.newaxis, 0,...]) 
            + 9 * (uc * VELOCITIES[:,0, jnp.newaxis, jnp.newaxis])) 
        + g[1] * (
            3 * (VELOCITIES[:, 1, jnp.newaxis, jnp.newaxis] - u[jnp.newaxis, 1,...]) 
            + 9 * (uc * VELOCITIES[:, 1, jnp.newaxis, jnp.newaxis])))
    
    return g_lattice


def get_source(g_lattice, omega):
    """Compute the source term caused by the forcing using Guo Forcing scheme.
    This term should be added to the DDF.
    
    Args:
        g_lattice (jax.Array of shape (9, NX, NY)): The discretize external force term.
        omega (scalar): The relaxation parameter (= 1 / relaxation time).
        
    Returns:
        out (jax.Array of shape (9, NX, NY)): The source term.
    """
   
    return g_lattice * (1 - 0.5 * omega)


# --------------------------------- boundary conditions ---------------------------------


def noslip_boundary(f, loc:str):
    """Enforce a no-slip boundary at the specified boundary using Bounce Back scheme. 
    
    Args:
        f (jax.Array): Discrete distribution function (DDF)
        loc (str): The boundary where the no-slip condition is enforced, 
            can be 'left', 'right', 'top', or 'bottom'.
        
    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF 
            after enforcing the boundary condition.
    """
    
    if loc == 'left':
        return f.at[RIGHT_DIRS, 0].set(f[LEFT_DIRS, 0])
    elif loc == 'right':
        return f.at[LEFT_DIRS, -1].set(f[RIGHT_DIRS, -1])
    elif loc == 'top':
        return f.at[DOWN_DIRS, :, -1].set(f[UP_DIRS, :, -1])
    elif loc == 'bottom':
        return f.at[UP_DIRS, :, 0].set(f[DOWN_DIRS, :, 0])
    else:
        raise ValueError("Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'.")


def noslip_obstacle(f, mask):
    """Enforce a no-slip boundary at the obstacle 
    using the Bounce Back scheme. The obstacle is defined by a 2D mask
    where True indicates the presence of an obstacle.
    
    Args:
        f (jax.Array): Discrete distribution function (DDF)
        mask (jax.Array of shape (NX, NY)): The mask indicating the obstacle.
    
    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF 
            after enforcing the boundary condition.    
    """
    
    return f.at[:, mask].set(f[:, mask][OPP_DIRS])


def velocity_boundary(f, ux, uy, loc:str):
    """Enforce given velocity ux, uy at the specified boundary using the
    Non-Equilibrium Bounce-Back (or Zou/He) scheme.
    
    Args:
        f (jax.Array): Discrete distribution function (DDF)
        ux (scalar or jax.Array of shape NX): The x-component of velocity.
        uy (scalar or jax.Array of shape NY): The y-component of velocity.
        loc (str): The boundary where the velocity condition is enforced, 
            can be 'left', 'right', 'top', or 'bottom'.
    
    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF 
            after enforcing the boundary condition.
    """
    
    if loc == 'left':        
        rho_wall = (f[0, 0] + f[2, 0] + f[4, 0] + 2 * (f[3, 0] + f[6, 0] + f[7, 0])) / (1 - ux)
        f = f.at[1, 0].set(f[3, 0] + 2 / 3 * ux * rho_wall)
        f = f.at[5, 0].set(f[7, 0] - 0.5 * (f[2, 0] - f[4, 0]) + (1 / 6 * ux + 0.5 * uy) * rho_wall)
        f = f.at[8, 0].set(f[6, 0] + 0.5 * (f[2, 0] - f[4, 0]) + (1 / 6 * ux - 0.5 * uy) * rho_wall)
    
    elif loc == 'right':        
        rho_wall = (f[0, -1] + f[2, -1] + f[4, -1] + 2 * (f[1, -1] + f[5, -1] + f[8, -1])) / (1 + ux)
        f = f.at[3, -1].set(f[1, -1] - 2 / 3 * ux * rho_wall)
        f = f.at[7, -1].set(f[5, -1] + 0.5 * (f[2, -1] - f[4, -1]) + (- 1 / 6 * ux - 0.5 * uy) * rho_wall)
        f = f.at[6, -1].set(f[8, -1] - 0.5 * (f[2, -1] - f[4, -1]) + (- 1 / 6 * ux + 0.5 * uy) * rho_wall)
    
    elif loc == 'top':        
        rho_wall = (f[0, :, -1] + f[1, :, -1] + f[3, :, -1] + 2 * (f[2, :, -1] + f[5, :, -1] + f[6, :, -1])) / (1 + uy)
        f = f.at[4, :, -1].set(f[2, :, -1] - 2 / 3 * uy * rho_wall)
        f = f.at[7, :, -1].set(f[5, :, -1] + 0.5 * (f[1, :, -1] - f[3, :, -1]) + (- 1 / 6 * uy - 0.5 * ux) * rho_wall)
        f = f.at[8, :, -1].set(f[6, :, -1] - 0.5 * (f[1, :, -1] - f[3, :, -1]) + (- 1 / 6 * uy + 0.5 * ux) * rho_wall)
    
    elif loc == 'bottom':        
        rho_wall = (f[0, :,0] + f[1, :,0] + f[3, :,0] + 2 * (f[4, :,0] + f[7, :,0] + f[8, :,0])) / (1 - uy)
        f = f.at[2, :, 0].set(f[4, :, 0] + 2 / 3 * uy * rho_wall)
        f = f.at[5, :, 0].set(f[7, :, 0] - 0.5 * (f[1, :, 0] - f[3, :, 0]) + (1 / 6 * uy + 0.5 * ux) * rho_wall)
        f = f.at[6, :, 0].set(f[8, :, 0] + 0.5 * (f[1, :, 0] - f[3, :, 0]) + (1 / 6 * uy - 0.5 * ux) * rho_wall)
    
    else:
        raise ValueError("Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'.")
    
    return f


def outlet_boundary(f, loc:str):
    """Enforce a no-gradient BC at the specified boundary using the Zou/He scheme.
    The outflow velocities is computed from the neighboring nodes.
    
    Args:
        f (jax.Array): Discrete distribution function (DDF)
        loc (str): The boundary where the outlet condition is enforced, 
            can be 'left', 'right', 'top', or 'bottom'.
            
    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF 
            after enforcing the boundary condition.
    """

    if loc == 'left':
        f_ = f[:, 0]    
    elif loc == 'right':
        f_ = f[:, -1]    
    elif loc == 'top':
        f_ = f[:, :, -1]
    elif loc == 'bottom':
        f_ = f[:, :, 0]
    else:
        raise ValueError("Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'.")
       
    rho_out = jnp.sum(f_, axis=0)
    ux_out = (jnp.sum(f_[RIGHT_DIRS], axis=0) - jnp.sum(f_[LEFT_DIRS], axis=0)) / rho_out
    uy_out = (jnp.sum(f_[UP_DIRS], axis=0) - jnp.sum(f_[DOWN_DIRS], axis=0)) / rho_out
    
    return velocity_boundary(f, ux_out, uy_out, loc)


def outlet_boundary_simple(f, loc:str):
    """Enforce a no-gradient outlet boundary at the specified boundary 
    by copying the second last row/column.
    
    Args:
        f (jax.Array): Discrete distribution function (DDF)
        loc (str): The boundary where the outlet condition is enforced, 
            can be 'left', 'right', 'top', or 'bottom'.
            
    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF 
            after enforcing the boundary condition.
    """

    if loc == 'left':
        return f.at[RIGHT_DIRS, 0].set(f[RIGHT_DIRS, 1])    
    elif loc == 'right':
        return f.at[LEFT_DIRS, -1].set(f[LEFT_DIRS, -2])
    elif loc == 'top':
        return f.at[DOWN_DIRS, :, -1].set(f[DOWN_DIRS, :, -2])
    elif loc == 'bottom':
        return f.at[UP_DIRS, :, 0].set(f[UP_DIRS, :, 1])
    else:
        raise ValueError("Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'.")


def boundary_equilibrium(f, feq, loc:str):
    """setting the missing distributions at boundary to the equilibrium value.
    
    Args:
        f (jax.Array): Discrete distribution function (DDF)
        feq (jax.Array of shape (9, NX, NY)): The equilibrium DDF,
            should has the same shape as the boundary.
        loc (str): The boundary where the outlet condition is enforced, 
            can be 'left', 'right', 'top', or 'bottom'.
    
    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF 
            after enforcing the boundary condition.    
    """
    
    if loc == 'left':
        return f.at[RIGHT_DIRS, 0].set(feq[RIGHT_DIRS])    
    elif loc == 'right':        
        return f.at[LEFT_DIRS, -1].set(feq[LEFT_DIRS])
    elif loc == 'top':
        return f.at[DOWN_DIRS, :, -1].set(feq[DOWN_DIRS])
    elif loc == 'bottom':
        return f.at[UP_DIRS, :, 0].set(feq[UP_DIRS])
    else:
        raise ValueError("Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'.")
