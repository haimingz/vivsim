"""This files provides the core functions for the lattice Boltzmann method (LBM) in 2D.

Collision models:
* Bhatnagar-Gross-Krook (BGK) model
* multiple relaxation time (MRT) model

Boundary conditions:
* No-slip boundary for domain boundaries and obstacles using the bounce-back scheme
* Velocity boundary for open boundaries using the Zou/He scheme 
* Non-reflection BC for open boundaries using Characteristic boundary condition (CBC) scheme 

All equations have been partially evaluated for the D2Q9 model to maximize efficiency.
"""

import numpy as np
import jax.numpy as jnp

def streaming(f):
    """Streaming fluid particles to their neighboring lattice nodes
    along their velocity directions.
    
    6   2   5
      \ | /
    3 - 0 - 1
      / | \
    7   4   8 

    Parameters:
    - f: The distribution function of shape (9, NX, NY).

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
    """Compu_wall_tes the fluid density and velocity according to the distribution functions.

    Parameters:
    - f: The distribution function of shape (9, NX, NY).
    - rho: The density of shape (NX, NY).
    - u: The velocity of shape (2, NX, NY).

    Returns:
    - rho: The density of shape (NX, NY).
    - u: The velocity of shape (2, NX, NY).
    """
    rho = jnp.sum(f, axis=0)
    u = u.at[0].set((f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho)
    u = u.at[1].set((f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho)
    return rho, u

def get_equilibrum(rho, u, feq):
    """Calculates the equilibrium distribution function.

    Parameters:
    - rho: The density of shape (NX, NY).
    - u: The velocity vector of shape (2, NX, NY).
    - feq: The equilibrium distribution function of shape (9, NX, NY).

    Returns:
    - feq: The updated equilibrium distribution function of shape (9, NX, NY).
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
    """Performs the collision step using the Bhatnagar-Gross-Krook (BGK) model.

    Parameters:
    - f: The distribution function of shape (9, NX, NY).
    - feq: The equilibrium distribution function of shape (9, NX, NY).
    - omega: The relaxation parameter.

    Returns:
    - The post-collision distribution function of shape (9, NX, NY).
    """
    return (1 - omega) * f + omega * feq

def get_omega_mrt(omega):
    """Generates the MRT omega matrix.

    Parameters:
    - omega: A scalar value representing the relaxation parameter.

    Returns:
    - The MRT omega matrix.
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
    """Performs collision step using the Multiple Relaxation Time (MRT) model.

    Args:
        f: The distribution function of shape (9, NX, NY).
        feq: The equilibrium distribution function of shape (9, NX, NY).
        omega_mrt: The relaxation rates for the MRT model of shape (9,9).

    Returns:
        ndarray: The updated distribution function after the collision step.
    """
    return jnp.tensordot(omega_mrt, feq - f, axes=([1], [0])) + f

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

def left_noslip_bb(f):
    """Applies no-slip BC at the left of the domain using the bounce back scheme.

    Parameters:
    - f: Distribution functions of shape (9, NX, NY).

    Returns:
    - f: Updated distribution functions after applying the boundary conditions.
    """

    f = f.at[right_indices, 0].set(f[left_indices, 0])
    return f

def right_noslip_bb(f):
    """Applies no-slip BC at the right of the domain using the bounce back scheme.

    Parameters:
    - f: Distribution functions of shape (9, NX, NY).

    Returns:
    - f: Updated distribution functions after applying the boundary conditions.
    """

    f = f.at[left_indices, -1].set(f[right_indices, -1])
    return f

def bottom_noslip_bb(f):
    """Applies no-slip BC at the bottom of the domain using the bounce back scheme.

    Parameters:
    - f: Distribution functions of shape (9, NX, NY).

    Returns:
    - f: Updated distribution functions after applying the boundary conditions.
    """

    f = f.at[top_indices, :, 0].set(f[bottom_indices, :, 0])
    return f

def top_noslip_bb(f):
    """Applies no-slip BC at the bottom of the domain using the bounce back scheme.

    Parameters:
    - f: Distribution functions of shape (9, NX, NY).

    Returns:
    - f: Updated distribution functions after applying the boundary conditions.
    """

    f = f.at[bottom_indices, :, -1].set(f[top_indices, :, -1])
    return f

def obj_noslip_bb(f, mask):
    """Applies no-slip BC at the object using the bounce back scheme.
    
    Parameters:
    - f: Distribution functions of shape (9, NX, NY).
    - mask: a matrix of shape (NX, NY) filled with 0 and 1 indicating null and the obj.
    
    Returns:
    - f: Updated distribution functions after applying the boundary conditions.
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


# Non-Equilibrium Bounce-Back (Zhou/He) scheme for open boundaries with given velocities

def left_velocity_nebb(f, rho, ux, uy):
    """
    Enforce given velocity at the left of the domain 
    using the Non-Equilibrium Bounce-Back (Zhou/He) scheme.

    Parameters:
    - f: Distribution functions of shape (9, NX, NY).
    - rho: Density of the fluid of shape (NX, NY).
    - ux, uy: inlet velocities as scalars or of the same shape as the boundary.

    Returns:
    - f: Updated distribution functions after applying the boundary conditions.
    - rho: Updated density after applying the boundary conditions.
    """

    rho_wall = (f[0, 0] + f[2, 0] + f[4, 0] + 2 * (f[3, 0] + f[6, 0] + f[7, 0])) / (- ux + 1)
    rho = rho.at[0].set(rho_wall)
    f = f.at[1, 0].set(f[3, 0] + 2 / 3 * ux * rho_wall)
    f = f.at[5, 0].set(f[7, 0] - 0.5 * (f[2, 0] - f[4, 0]) + (1 / 6 * ux + 0.5 * uy) * rho_wall)
    f = f.at[8, 0].set(f[6, 0] + 0.5 * (f[2, 0] - f[4, 0]) + (1 / 6 * ux - 0.5 * uy) * rho_wall)
    return f, rho

def right_velocity_nebb(f, rho, ux, uy):
    """
    Enforce given velocity at the right of the domain 
    using the Non-Equilibrium Bounce-Back (Zhou/He) scheme.
    
    Parameters:
    - f: Distribution functions of shape (9, NX, NY).
    - rho: Density of the fluid of shape (NX, NY).
    - ux, uy: inlet velocities as scalars or of the same shape as the boundary.
    
    Returns:
    - f: Updated distribution functions after applying the boundary conditions.
    - rho: Updated density after applying the boundary conditions.
    """
    
    rho_wall = (f[0, -1] + f[2, -1] + f[4, -1] + 2 * (f[1, -1] + f[5, -1] + f[8, -1])) / (ux + 1)
    rho = rho.at[-1].set(rho_wall)
    f = f.at[3, -1].set(f[1, -1] - 2 / 3 * ux * rho_wall)
    f = f.at[7, -1].set(f[5, -1] + 0.5 * (f[2, -1] - f[4, -1]) + (- 1 / 6 * ux - 0.5 * uy) * rho_wall)
    f = f.at[6, -1].set(f[8, -1] - 0.5 * (f[2, -1] - f[4, -1]) + (- 1 / 6 * ux + 0.5 * uy) * rho_wall)
    return f, rho

def top_velocity_nebb(f, rho, ux, uy):
    """
    Enforce given velocity at the top of the domain 
    using the Non-Equilibrium Bounce-Back (Zhou/He) scheme.
    
    Parameters:
    - f: Distribution functions of shape (9, NX, NY).
    - rho: Density of the fluid of shape (NX, NY).
    - ux, uy: inlet velocities as scalars or of the same shape as the boundary.
    
    Returns:
    - f: Updated distribution functions after applying the boundary conditions.
    - rho: Updated density after applying the boundary conditions.
    """
    
    rho_wall = (f[0, :,-1] + f[1, :,-1] + f[3, :,-1] + 2 * (f[2, :,-1] + f[5, :,-1] + f[6, :,-1])) / (uy + 1)
    rho = rho.at[:, -1].set(rho_wall)
    f = f.at[4, :, -1].set(f[2, :, -1] - 2 / 3 * uy * rho_wall)
    f = f.at[7, :, -1].set(f[5, :, -1] + 0.5 * (f[1, :, -1] - f[3, :, -1]) + (1 / 6 * uy - 0.5 * ux) * rho_wall)
    f = f.at[8, :, -1].set(f[6, :, -1] - 0.5 * (f[1, :, -1] - f[3, :, -1]) + (1 / 6 * uy + 0.5 * ux) * rho_wall)
    return f, rho

def bottom_velocity_nebb(f, rho, ux, uy):
    """
    Enforce given velocity at the bottom of the domain 
    using the Non-Equilibrium Bounce-Back (Zhou/He) scheme.
    
    Parameters:
    - f: Distribution functions of shape (9, NX, NY).
    - rho: Density of the fluid of shape (NX, NY).
    - ux, uy: inlet velocities as scalars or of the same shape as the boundary.
    
    Returns:
    - f: Updated distribution functions after applying the boundary conditions.
    - rho: Updated density after applying the boundary conditions.
    """
    
    rho_wall = (f[0, :,0] + f[1, :,0] + f[3, :,0] + 2 * (f[4, :,0] + f[7, :,0] + f[8, :,0])) / (- uy + 1)
    rho = rho.at[:, 0].set(rho_wall)
    f = f.at[2, :, 0].set(f[4, :, 0] + 2 / 3 * uy * rho_wall)
    f = f.at[5, :, 0].set(f[7, :, 0] - 0.5 * (f[1, :, 0] - f[3, :, 0]) + (1 / 6 * uy + 0.5 * ux) * rho_wall)
    f = f.at[6, :, 0].set(f[8, :, 0] + 0.5 * (f[1, :, 0] - f[3, :, 0]) + (1 / 6 * uy - 0.5 * ux) * rho_wall)
    return f, rho


# Apply no gradient BC for ou_wall_tflow by simply copying the second last row/column (1st order accuracy)

def right_outlet_simple(f):
    """Applies outlet BC at the right of the domain (No gradient BC).

    Parameters:
    - f: Distribution functions of shape (9, NX, NY).

    Returns:
    - f: Updated distribution functions after applying the boundary conditions.
    """
    f = f.at[left_indices, -1].set(f[left_indices, -2])
    return f

def left_outlet_simple(f):
    """Applies outlet BC at the left of the domain (No gradient BC).

    Parameters:
    - f: Distribution functions of shape (9, NX, NY).

    Returns:
    - f: Updated distribution functions after applying the boundary conditions.
    """

    f = f.at[right_indices, 0].set(f[right_indices, 1])
    return f

def top_outlet_simple(f):
    """Applies outlet BC at the top of the domain (No gradient BC).

    Parameters:
    - f: Distribution functions of shape (9, NX, NY).

    Returns:
    - f: Updated distribution functions after applying the boundary conditions.
    """

    f = f.at[top_indices, :, -1].set(f[top_indices, :, -2])
    return f

def bottom_outlet_simple(f):
    """Applies outlet BC at the bottom of the domain (No gradient BC).

    Parameters:
    - f: Distribution functions of shape (9, NX, NY).

    Returns:
    - f: Updated distribution functions after applying the boundary conditions.
    """

    f = f.at[bottom_indices, :, 0].set(f[bottom_indices, :, 1])
    return f


# Characteristic boundary condition (CBC) for non-reflection boundaries

def right_cbc(rho, u):
    """characteristic boundary condition for the right BC
    
    Parameters:
    - rho: The density of shape (NX, NY).
    - u: The velocity of shape (2, NX, NY).
    
    Returns:
    - rhob_next: The boundary density for the next time step.
    - ub_next: The boundary velocity for the next time step.
    """

    uxb = u[0, -1]
    uyb = u[1, -1]
    rhob = rho[-1]
    
    # spatial derivatives using finite difference
    uxxb = (3 * uxb - 4 * u[0, -2] + u[0, -3]) / 2
    uyxb = (3 * uyb - 4 * u[1, -2] + u[1, -3]) / 2
    rhoxb = (3 * rhob - 4 * rho[-2] + rho[-3]) / 2
    uxyb = jnp.gradient(uxb)
    uyyb = jnp.gradient(uyb)
    rhoyb = jnp.gradient(rhob)
    
    # characteristic variables
    cs = 1 / jnp.sqrt(3) # the speed of sound
    cs2 = 1 / 3  # cs^2
    lx1 = (uxb - cs) * (cs2 * rhoxb - cs * rhob * uxxb)
    lx2 = uxb * uyxb
    lx3 = (uxb + cs) * (cs2 * rhoxb + cs * rhob * uxxb)
    
    # evaluate the time derivatives using modified Thompson's scheme
    rhotb = - 1.5 * lx1 - 0.75 * (uyb * rhoyb + rhob * uyyb)
    utb = jnp.zeros_like(u[:,-1])
    utb = utb.at[0].set(+ 0.5 / cs / rhob * lx1 - 0.75 * uyb * uxyb)
    utb = utb.at[1].set(- lx2 - 0.75 * (cs2 / rhob * rhoyb + uyb * uyyb))
    
    # calculate next time step values using Buler method
    rhob_next = rho[-1] + rhotb
    ub_next = u[:, -1] + utb
    
    return rhob_next, ub_next


def right_cbc_old(rho, u):
    """characteristic boundary condition for the right BC"""

    uxx = (3 * u[0,-1] - 4 * u[0, -2] + u[0, -3]) / 2
    uyx = (3 * u[1,-1] - 4 * u[1, -2] + u[1, -3]) / 2
    rhox = (3 * rho[-1] - 4 * rho[-2] + rho[-3]) / 2
    uyy = jnp.gradient(u[1, -1])
    uxy = jnp.gradient(u[0, -1])
    rhoy = jnp.gradient(rho[-1])
    
    lx1 = (u[0, -1] - 1 / jnp.sqrt(3)) * (1 / 3 * rhox - 1 / jnp.sqrt(3) * rho[-1] * uxx)
    lx2 = u[0, -1] * uyx
    lx3 = (u[0, -1] + 1 / jnp.sqrt(3)) * (1 / 3 * rhox + 1 / jnp.sqrt(3) * rho[-1] * uxx)
    
    rhot = -1.5 * lx1 - u[1,-1] * rhoy - rho[-1] * uyy    
    uxt = 0.5 * jnp.sqrt(3) / rho[-1] * lx1 - u[1,-1] * uxy 
    uyt = - lx2 - 1 / 3 / rho[-1] * rhoy - u[1,-1] * uyy
    
    ut = jnp.zeros_like(u[:,-1])
    ut = ut.at[0].set(uxt)
    ut = ut.at[1].set(uyt)
    
    return rhot, ut