"""This file implements the 2D lattice Boltzmann method (LBM) using the D2Q9 model.

The implementation includes:

Collision Model:
* Bhatnagar-Gross-Krook (BGK) model, also known as Single Relaxation Time (SRT) model

Boundary Conditions:
* Periodic: Natural wrapping at domain boundaries
* No-slip: Bounce-back scheme for walls and obstacles 
* Velocity (Dirichlet): NEBB/Zou-He scheme for prescribed velocity at boundaries
* Outflow (Neumann): Zero-gradient via first-order extrapolation

Force Implementation:
* Guo forcing scheme for external forces

D2Q9 Lattice Structure:
The velocity space is discretized into 9 directions, numbered as follows:

    6   2   5
      \ | /
    3 - 0 - 1
      / | \
    7   4   8

Key Variables:
    f: Distribution functions, shape (9, NX, NY)
    feq: Equilibrium distribution functions, shape (9, NX, NY) 
    rho: Macroscopic density, shape (NX, NY)
    u: Macroscopic velocity vector, shape (2, NX, NY)
    g: External force vector, shape (2, NX, NY)

All equations are pre-evaluated for the D2Q9 model for maximum efficiency.
"""

import jax.numpy as jnp

def streaming(f):
    """Perform the streaming step of the LBM by shifting the distribution functions 
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
    """Compute the macroscopic fluid density and velocity."""
    
    rho = jnp.sum(f, axis=0)
    u = u.at[0].set((f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho)
    u = u.at[1].set((f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho)
    return rho, u

def get_equilibrium(rho, u, feq):
    """Calculate the equilibrium distribution function."""

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
    """Perform the collision step using the Bhatnagar-Gross-Krook (BGK) model
    where omega is the relaxation parameter. """

    return (1 - omega) * f + omega * feq

def get_discretized_force(g, u):
    """Calculate the discretized force from the macroscopic velocity and external force.
    
    The discretized force (gd) has the same shape as the distribution functions (f).
    """
    
    gxux = g[0] * u[0]
    gyuy = g[1] * u[1]
    gxuy = g[0] * u[1]
    gyux = g[1] * u[0]
    
    gd = jnp.zeros((9, u.shape[1], u.shape[2]))
    gd = gd.at[0].set(4 / 3 * (- gxux - gyuy))
    gd = gd.at[1].set(1 / 3 * (2 * gxux + g[0] - gyuy))
    gd = gd.at[2].set(1 / 3 * (2 * gyuy + g[1] - gxux))
    gd = gd.at[3].set(1 / 3 * (2 * gxux - g[0] - gyuy))
    gd = gd.at[4].set(1 / 3 * (2 * gyuy - g[1] - gxux))
    gd = gd.at[5].set(1 / 12 * (2 * gxux + 3 * gxuy + g[0] + 3 * gyux + 2 * gyuy + g[1]))
    gd = gd.at[6].set(1 / 12 * (2 * gxux - 3 * gxuy - g[0] - 3 * gyux + 2 * gyuy + g[1]))
    gd = gd.at[7].set(1 / 12 * (2 * gxux + 3 * gxuy - g[0] + 3 * gyux + 2 * gyuy - g[1]))
    gd = gd.at[8].set(1 / 12 * (2 * gxux - 3 * gxuy + g[0] - 3 * gyux + 2 * gyuy - g[1]))
    
    return gd

def get_velocity_correction(g, rho=1):
    """Compute the velocity correction due to the force term."""

    return g * 0.5 / rho

def get_source(gd, omega):
    """Compute the source term needed to be added to the distribution functions
    according to Guo's scheme."""
   
    return gd * (1 - 0.5 * omega)


# --------------------------------- boundary conditions ---------------------------------

right_indices = jnp.array([1, 5, 8])
left_indices = jnp.array([3, 7, 6])
top_indices = jnp.array([2, 5, 6])
bottom_indices = jnp.array([4, 7, 8])
all_indices = jnp.array([0,1,2,3,4,5,6,7,8])
opposite_indices = jnp.array([0, 3,4,1,2,7,8,5,6])


# Bounce-back scheme for no-slip boundaries

def left_noslip(f):
    """Enforce a no-slip boundary at the left of the domain using the Bounce Back scheme."""

    return f.at[right_indices, 0].set(f[left_indices, 0])

def right_noslip(f):
    """Enforce a no-slip boundary at the right of the domain using the Bounce Back scheme. """

    return f.at[left_indices, -1].set(f[right_indices, -1])

def bottom_noslip(f):
    """Enforce a no-slip boundary at the bottom of the domain using the Bounce Back scheme."""

    return f.at[top_indices, :, 0].set(f[bottom_indices, :, 0])

def top_noslip(f):
    """Enforce a no-slip boundary at the bottom of the domain using the Bounce Back scheme."""

    return f.at[bottom_indices, :, -1].set(f[top_indices, :, -1])

def obstacle_noslip(f, mask):
    """Enforce a no-slip boundary at the obstacle 
    using the Bounce Back scheme. The obstacle is defined by a 2D mask
    where True indicates the presence of an obstacle."""
    
    return f.at[:, mask].set(f[:, mask][opposite_indices])


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
    
    return f.at[left_indices, -1].set(f[left_indices, -2])

def left_outflow(f):
    """Enforce an outflow boundary at the left of the domain
    by just copying the second last row/column."""

    return f.at[right_indices, 0].set(f[right_indices, 1])

def top_outflow(f):
    """Enforce an outflow boundary at the top of the domain
    by just copying the second last row/column."""

    return f.at[top_indices, :, -1].set(f[top_indices, :, -2])

def bottom_outflow(f):
    """Enforce an outflow boundary at the bottom of the domain
    by just copying the second last row/column."""

    return f.at[bottom_indices, :, 0].set(f[bottom_indices, :, 1])
