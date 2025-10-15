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

Key Variables:
    * rho: Macroscopic density, shape (NX, NY)
    * u: Macroscopic velocity vector, shape (2, NX, NY)
    * f: Discrete Distribution Function (DDF), shape (9, NX, NY)
    * feq: Equilibrium DDF, shape (9, NX, NY) 
    * omega: Relaxation parameter, scalar
    * nu: Kinematic viscosity in lattice units, scalar
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
    """Perform the streaming step of the Lattice Boltzmann Method.
    
    This function shifts the distribution functions along their respective velocity
    directions using periodic boundary conditions. Each population is propagated to
    its neighboring node according to the D2Q9 lattice velocities.
    
    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        
    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF after streaming.
    """
    
    f = f.at[RIGHT_DIRS].set(jnp.roll(f[RIGHT_DIRS], 1, axis=1))
    f = f.at[LEFT_DIRS].set(jnp.roll(f[LEFT_DIRS], -1, axis=1))
    f = f.at[UP_DIRS].set(jnp.roll(f[UP_DIRS], 1, axis=2))
    f = f.at[DOWN_DIRS].set(jnp.roll(f[DOWN_DIRS], -1, axis=2))
    return f


def get_macroscopic(f):
    """Calculate macroscopic properties from the distribution function.
    
    Computes the fluid density and velocity by taking moments of the distribution
    function. The density is the zeroth moment (sum of all populations), and the
    velocity is derived from the first moment (momentum divided by density).
    
    Note: When body forces are present, use get_velocity_correction() to adjust
    the velocity for second-order accuracy.
    
    Args:
        f (jax.Array of shape (9, NX, NY) or (9, NX) or (9, NY)): 
            Discrete distribution function (DDF). Can be a full 2D domain or 1D slice.
    
    Returns:
        rho (jax.Array of shape (NX, NY) or (NX,) or (NY,)): The macroscopic density.
        u (jax.Array of shape (2, NX, NY) or (2, NX) or (2, NY)): The macroscopic velocity.
    """
    
    rho = jnp.sum(f, axis=0)
    u = jnp.zeros((2, *rho.shape))
    u = u.at[0].set((jnp.sum(f[RIGHT_DIRS], axis=0) - jnp.sum(f[LEFT_DIRS], axis=0)) / rho)
    u = u.at[1].set((jnp.sum(f[UP_DIRS], axis=0) - jnp.sum(f[DOWN_DIRS], axis=0)) / rho)
    return rho, u


def get_equilibrium(rho, u):
    """Compute the equilibrium distribution function from macroscopic properties.
    
    Calculates the Maxwell-Boltzmann equilibrium distribution for the D2Q9 lattice
    using the second-order expansion. The equilibrium is a function of density and
    velocity that the distribution function relaxes toward during collision.
    
    Args:
        rho (jax.Array of shape (*spatial_dims)): The macroscopic density.
            Can be a scalar, 1D array (NX,) or (NY,), or 2D array (NX, NY).
        u (jax.Array of shape (2, *spatial_dims)): The macroscopic velocity.
            First dimension is the velocity components (x, y).

    Returns:
        feq (jax.Array of shape (9, *spatial_dims)): The equilibrium DDF.
    """
    
    ndim = len(rho.shape)
    uc = jnp.sum(u[jnp.newaxis, ...] *  VELOCITIES.reshape((9, 2) + (1,) * ndim), axis=1)
    feq = (rho * WEIGHTS.reshape((9,) + (1,) * ndim) * 
          (1 + 3 * uc + 4.5 * uc ** 2 - 1.5 * jnp.sum(u ** 2, axis=0)))
    return feq 


def collision_bgk(f, feq, omega):
    """Perform the collision step using the Bhatnagar-Gross-Krook (BGK) model.
    
    Also known as the Single-Relaxation-Time (SRT) model, this collision operator
    relaxes the distribution function toward its equilibrium state. The relaxation
    parameter omega controls the rate of relaxation and is related to the fluid
    viscosity.
    
    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        feq (jax.Array of shape (9, NX, NY)): The equilibrium DDF.
        omega (scalar): The relaxation parameter, omega = 1 / tau, where tau is
            the relaxation time. Related to viscosity by: nu = (1/omega - 0.5) / 3.
    
    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF after collision.
    """

    return (1 - omega) * f + omega * feq


def get_omega(nu):
    """Compute the relaxation parameter from the kinematic viscosity.
    
    Converts the physical kinematic viscosity (in lattice units) to the BGK
    relaxation parameter omega used in the collision step. This relationship
    comes from the Chapman-Enskog expansion of the LBM equations.
    
    Args:
        nu (scalar): The kinematic viscosity in lattice units.
        
    Returns:
        omega (scalar): The relaxation parameter, omega = 1 / (3*nu + 0.5).
    """
    return 1 / (3 * nu + 0.5)


def get_velocity_correction(g, rho=1):
    """Compute the velocity correction in the presence of external body force. 
    
    The result should be added to the fluid velocity obtained from `get_macroscopic`
    to preserve second-order accuracy when body forces are present. This correction
    is independent of the forcing scheme used.
    
    Args:
        g (scalar or jax.Array): The external body force density. 
            Can be a scalar or array of shape (2, NX, NY) for full domain,
            or (NX,) or (NY,) for boundary slices, or a single scalar value.
        rho (scalar or jax.Array): The macroscopic density.
            Can be a scalar or array with compatible shape to g.
        
    Returns:
        out (scalar or jax.Array): The velocity correction with the same shape as g.
            Should be added to the velocity: u_corrected = u + get_velocity_correction(g, rho)
    """

    return g * 0.5 / rho

