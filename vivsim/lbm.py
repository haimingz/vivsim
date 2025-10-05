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
        rho (jax.Array): The macroscopic density with shape (*spatial_dims).
        u (jax.Array): The macroscopic velocity with shape (2, *spatial_dims).
        They can be either 1D slice or 2D block.

    Returns:
        feq (jax.Array): The equilibrium DDF with shape (9, *spatial_dims).    
    """
    
    ndim = len(rho.shape)
    uc = jnp.sum(u[jnp.newaxis, ...] *  VELOCITIES.reshape((9, 2) + (1,) * ndim), axis=1)
    feq = (rho * WEIGHTS.reshape((9,) + (1,) * ndim) * 
          (1 + 3 * uc + 4.5 * uc ** 2 - 1.5 * jnp.sum(u ** 2, axis=0)))
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


def noslip_boundary(f, loc:str, scheme='nebb'):
    """Enforce a no-slip boundary at the specified boundary. 
    
    This is a wrapper function that calls either the `nebb_velocity` for
    Non-Equilibrium Bounce Back (NEBB) scheme (default) or the `nee_velocity` for
    Non-Equilibrium Extrapolation (NEE) scheme. Directly calling these two functions
    are recommended. This can be removed in future versions for consistency.

    Args:
        f (jax.Array): Discrete distribution function (DDF)
        loc (str): The boundary where the no-slip condition is enforced, 
            can be 'left', 'right', 'top', or 'bottom'.
        scheme (str): The scheme used to enforce the no-slip condition,
            can be 'nebb' (Non-Equilibrium Bounce Back) or 'nee' (Non-Equilibrium Extrapolation).

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF 
            after enforcing the boundary condition.
    """
    
    if scheme == 'nebb':
        return nebb_velocity(f, loc, ux_wall=0, uy_wall=0)
    if scheme == 'nee':
        return nee_velocity(f, loc, ux_wall=0, uy_wall=0)


def noslip_obstacle(f, mask):
    """Enforce a no-slip boundary at the obstacle using the Bounce Back scheme. 
    
    This is for backward compatibility. Use `bounce_back_obstacle` instead.
    This can be removed in future versions for consistency.
    
    Args:
        f (jax.Array): Discrete distribution function (DDF)
        mask (jax.Array of shape (NX, NY)): The mask indicating the obstacle.
    
    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF 
            after enforcing the boundary condition.    
    """
    
    return bounce_back_obstacle(f, mask)


def velocity_boundary(f, ux, uy, loc:str, scheme='nebb'):
    """Enforce given velocity ux, uy at the specified boundary.
    
    This is a wrapper function that calls either the `nebb_velocity` for
    Non-Equilibrium Bounce Back (NEBB) scheme or the `nee_velocity` for
    Non-Equilibrium Extrapolation (NEE) scheme. Directly calling these two functions
    are recommended. This can be removed in future versions for consistency.

    Args:
        f (jax.Array): Discrete distribution function (DDF)
        ux (scalar or jax.Array of shape NX): The x-component of velocity.
        uy (scalar or jax.Array of shape NY): The y-component of velocity.
        loc (str): The boundary where the velocity condition is enforced, 
            can be 'left', 'right', 'top', or 'bottom'.
        scheme (str): The scheme used to enforce the velocity boundary,
            can be 'nebb' (Non-Equilibrium Bounce Back) or 'nee' (Non-Equilibrium Extrapolation).

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF 
            after enforcing the boundary condition.
    """

    if scheme == 'nebb':
        return nebb_velocity(f, loc, ux_wall=ux, uy_wall=uy)
    if scheme == 'nee':
        return nee_velocity(f, loc, ux_wall=ux, uy_wall=uy)


def outlet_boundary(f, loc:str, scheme='nebb'):
    """Enforce a no pressure outlet BC at the specified boundary.
    
    This is a wrapper function that calls either the `nebb_pressure` for
    Non-Equilibrium Bounce Back (NEBB) scheme or the `nee_pressure` for
    Non-Equilibrium Extrapolation (NEE) scheme. Directly calling these two functions
    are recommended. This can be removed in future versions for consistency.

    Args:
        f (jax.Array): Discrete distribution function (DDF)
        loc (str): The boundary where the outlet condition is enforced, 
            can be 'left', 'right', 'top', or 'bottom'.
        scheme (str): The scheme used to enforce the outlet boundary,
            can be 'nebb' (Non-Equilibrium Bounce Back) or 'nee' (Non-Equilibrium Extrapolation).

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF 
            after enforcing the boundary condition.
    """

    if scheme == 'nebb':
        return nebb_pressure(f, loc, rho_wall=1)
    if scheme == 'nee':
        return nee_pressure(f, loc, rho_wall=1)


def outlet_boundary_simple(f, loc:str):
    """Removed due to poor accuracy. Use `outlet_boundary` instead."""

    raise ValueError("Function `outlet_boundary_simple` is removed due to poor accuracy. Use `nebb_pressure` or `nee_pressure` instead for outlet boundary conditions.")


def boundary_equilibrium(f, feq, loc:str):
    """Removed due to poor accuracy. Use `outlet_boundary` instead."""

    raise ValueError("Function `boundary_equilibrium` is removed due to poor accuracy. Use `nebb_pressure` or `nee_pressure` instead for outlet boundary conditions.")



def bounce_back(f_before_stream, f, loc, ux_wall=0, uy_wall=0):
    """Apply no-slip boundary using the bounce-back method on the specified boundary.

    This function should be called after the streaming step.

    In the streaming step, the outgoing distribution functions have been moved to the
    opposite side of the domain due to the `jnp.roll` operation. However, there is a great
    chance that these distribution functions have been modified by other boundary conditions
    before this function is called. Therefore, we need to directly obtain that information 
    from the distribution functions before the streaming step. This can be done by assigning
    the output of `streaming` to a new variable, e.g., `f_new = streaming(f)` for later use 
    and keep the original `f` unchanged.

    In this function, we will reverse the directions of the outgoing distribution functions
    in-place. When the no-slip boundary is moving, we also need to account for the
    momentum exchange due to the moving boundary.

    Args:
        f (jax.Array): The DDF after streaming, shape (9, NX, NY).
        f_before_stream (jax.Array): The DDF before streaming, shape (9, NX, NY).
        loc (str): The location of the boundary where the bounce-back is applied.
                   Should be one of 'left', 'right', 'top', or 'bottom'.
        ux_wall (scalar or jax.Array of shape NX): The x-component of the wall velocity.
        uy_wall (scalar or jax.Array of shape NY): The y-component of the wall velocity.

    Returns:
        f (jax.Array): The DDF after applying the bounce-back boundary condition, shape (9, NX, NY).
    """

    if loc not in ["left", "right", "top", "bottom"]:
        raise ValueError(
            "Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'."
        )

    if loc == "left":
        f = f.at[1, 0].set(f_before_stream[3, 0] + 2 / 3 * ux_wall)
        f = f.at[5, 0].set(f_before_stream[7, 0] - 1 / 6 * (-ux_wall - uy_wall))
        f = f.at[8, 0].set(f_before_stream[6, 0] - 1 / 6 * (-ux_wall + uy_wall))

    elif loc == "right":
        f = f.at[3, -1].set(f_before_stream[1, -1] - 2 / 3 * ux_wall)
        f = f.at[7, -1].set(f_before_stream[5, -1] - 1 / 6 * (ux_wall + uy_wall))
        f = f.at[6, -1].set(f_before_stream[8, -1] - 1 / 6 * (ux_wall - uy_wall))

    elif loc == "top":
        f = f.at[4, :, -1].set(f_before_stream[2, :, -1] - 2 / 3 * uy_wall)
        f = f.at[7, :, -1].set(f_before_stream[5, :, -1] - 1 / 6 * (ux_wall + uy_wall))
        f = f.at[8, :, -1].set(f_before_stream[6, :, -1] - 1 / 6 * (-ux_wall + uy_wall))

    elif loc == "bottom":
        f = f.at[2, :, 0].set(f_before_stream[4, :, 0] + 2 / 3 * uy_wall)
        f = f.at[5, :, 0].set(f_before_stream[7, :, 0] - 1 / 6 * (-ux_wall - uy_wall))
        f = f.at[6, :, 0].set(f_before_stream[8, :, 0] - 1 / 6 * (ux_wall - uy_wall))

    return f


def specular_reflection(f_before_stream, f, loc, ux_wall=0, uy_wall=0):
    """Apply slip boundary using the specular reflection method on the specified boundary.

    This function should be called after the streaming step.

    In the streaming step, the outgoing distribution functions have been moved to the
    opposite side of the domain due to the `jnp.roll` operation. However, there is a great
    chance that these distribution functions have been modified by other boundary conditions
    before this function is called. Therefore, we need to directly obtain that information 
    from the distribution functions before the streaming step. This can be done by assigning
    the output of `streaming` to a new variable, e.g., `f_new = streaming(f)` for later use 
    and keep the original `f` unchanged.

    The difference between specular reflection and bounce-back is that in specular reflection,
    the directions of the hit-wall particles are not reversed, but reflected like a mirror.

    Args:
        f (jax.Array): The DDF after streaming, shape (9, NX, NY).
        f_before_stream (jax.Array): The DDF before streaming, shape (9, NX, NY).
        loc (str): The location of the boundary where the bounce-back is applied.
                   Should be one of 'left', 'right', 'top', or 'bottom'.
        ux_wall (scalar or jax.Array of shape NX): The x-component of the wall velocity.
        uy_wall (scalar or jax.Array of shape NY): The y-component of the wall velocity.

    Returns:
        f (jax.Array): The DDF after applying the bounce-back boundary condition, shape (9, NX, NY).
    """

    if loc not in ["left", "right", "top", "bottom"]:
        raise ValueError(
            "Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'."
        )

    if loc == "left":
        f = f.at[1, 0].set(f_before_stream[3, 0] + 2 / 3 * ux_wall)
        f = f.at[8, 0].set(f_before_stream[7, 0] - 1 / 6 * (-ux_wall - uy_wall))
        f = f.at[5, 0].set(f_before_stream[6, 0] - 1 / 6 * (-ux_wall + uy_wall))

    elif loc == "right":
        f = f.at[3, -1].set(f_before_stream[1, -1] - 2 / 3 * ux_wall)
        f = f.at[6, -1].set(f_before_stream[5, -1] - 1 / 6 * (ux_wall + uy_wall))
        f = f.at[7, -1].set(f_before_stream[8, -1] - 1 / 6 * (ux_wall - uy_wall))

    elif loc == "top":
        f = f.at[4, :, -1].set(f_before_stream[2, :, -1] - 2 / 3 * uy_wall)
        f = f.at[8, :, -1].set(f_before_stream[5, :, -1] - 1 / 6 * (ux_wall + uy_wall))
        f = f.at[7, :, -1].set(f_before_stream[6, :, -1] - 1 / 6 * (-ux_wall + uy_wall))

    elif loc == "bottom":
        f = f.at[2, :, 0].set(f_before_stream[4, :, 0] + 2 / 3 * uy_wall)
        f = f.at[6, :, 0].set(f_before_stream[7, :, 0] - 1 / 6 * (-ux_wall - uy_wall))
        f = f.at[5, :, 0].set(f_before_stream[8, :, 0] - 1 / 6 * (ux_wall - uy_wall))

    return f


def bounce_back_obstacle(f, mask):
    """Enforce a no-slip boundary using the Bounce Back scheme
    at any obstacles defined by a 2D mask, where True indicates
    the presence of an obstacle. 
    
    This can also be used to enforce no-slip boundarys at domain boundaries.

    Args:
        f (jax.Array): Discrete distribution function (DDF)
        mask (jax.Array of shape (NX, NY)): The mask indicating the obstacle.

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """

    return f.at[:, mask].set(f[:, mask][OPP_DIRS])


def _nebb(f, loc: str, ux_wall=0, uy_wall=0, rho_wall=1):
    """Non-Equilibrium Bounce-Back (or Zou/He) scheme.

    This is the core function for the nebb_velocity and nebb_pressure functions.

    It applies the Non-Equilibrium Bounce-Back (or Zou/He) scheme to the distribution function
    at the specified boundary location, enforcing the given wall velocity and density.
    It only modifies the unknown incoming distribution functions at the boundary, while keeping
    the known outgoing distribution functions unchanged.

    Args:
        f (jax.Array): Discrete distribution function (DDF)
        loc (str): The boundary where the velocity condition is enforced,
            can be 'left', 'right', 'top', or 'bottom'.
        ux_wall (scalar or jax.Array of shape NX): The x-component of velocity.
        uy_wall (scalar or jax.Array of shape NY): The y-component of velocity.
        rho_wall (scalar or jax.Array of shape NX or NY): The density at the wall.

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """

    if loc == "left":
        f = f.at[1, 0].set(f[3, 0] + 2 / 3 * ux_wall * rho_wall)
        f = f.at[5, 0].set(
            f[7, 0]
            - 0.5 * (f[2, 0] - f[4, 0])
            + (1 / 6 * ux_wall + 0.5 * uy_wall) * rho_wall
        )
        f = f.at[8, 0].set(
            f[6, 0]
            + 0.5 * (f[2, 0] - f[4, 0])
            + (1 / 6 * ux_wall - 0.5 * uy_wall) * rho_wall
        )

    elif loc == "right":
        f = f.at[3, -1].set(f[1, -1] - 2 / 3 * ux_wall * rho_wall)
        f = f.at[7, -1].set(
            f[5, -1]
            + 0.5 * (f[2, -1] - f[4, -1])
            + (-1 / 6 * ux_wall - 0.5 * uy_wall) * rho_wall
        )
        f = f.at[6, -1].set(
            f[8, -1]
            - 0.5 * (f[2, -1] - f[4, -1])
            + (-1 / 6 * ux_wall + 0.5 * uy_wall) * rho_wall
        )

    elif loc == "top":
        f = f.at[4, :, -1].set(f[2, :, -1] - 2 / 3 * uy_wall * rho_wall)
        f = f.at[7, :, -1].set(
            f[5, :, -1]
            + 0.5 * (f[1, :, -1] - f[3, :, -1])
            + (-1 / 6 * uy_wall - 0.5 * ux_wall) * rho_wall
        )
        f = f.at[8, :, -1].set(
            f[6, :, -1]
            - 0.5 * (f[1, :, -1] - f[3, :, -1])
            + (-1 / 6 * uy_wall + 0.5 * ux_wall) * rho_wall
        )

    elif loc == "bottom":
        f = f.at[2, :, 0].set(f[4, :, 0] + 2 / 3 * uy_wall * rho_wall)
        f = f.at[5, :, 0].set(
            f[7, :, 0]
            - 0.5 * (f[1, :, 0] - f[3, :, 0])
            + (1 / 6 * uy_wall + 0.5 * ux_wall) * rho_wall
        )
        f = f.at[6, :, 0].set(
            f[8, :, 0]
            + 0.5 * (f[1, :, 0] - f[3, :, 0])
            + (1 / 6 * uy_wall - 0.5 * ux_wall) * rho_wall
        )

    return f


def nebb_velocity(f, loc: str, ux_wall=0, uy_wall=0):
    """Enforce given velocity ux_wall, uy_wall at the specified boundary using the
    Non-Equilibrium Bounce-Back (or Zou/He) scheme.

    This function should be called after the streaming step. In this function,
    the density at the wall is computed based on the known outgoing distribution functions.

    Args:
        f (jax.Array): Discrete distribution function (DDF)
        loc (str): The boundary where the velocity condition is enforced,
            can be 'left', 'right', 'top', or 'bottom'.
        ux_wall (scalar or jax.Array of shape NX): The x-component of velocity.
        uy_wall (scalar or jax.Array of shape NY): The y-component of velocity.

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """
    if loc not in ["left", "right", "top", "bottom"]:
        raise ValueError(
            "Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'."
        )

    if loc == "left":
        rho_wall = (f[0, 0] + f[2, 0] + f[4, 0] + 2 * (f[3, 0] + f[6, 0] + f[7, 0])) / (
            1 - ux_wall
        )

    elif loc == "right":
        rho_wall = (
            f[0, -1] + f[2, -1] + f[4, -1] + 2 * (f[1, -1] + f[5, -1] + f[8, -1])
        ) / (1 + ux_wall)

    elif loc == "top":
        rho_wall = (
            f[0, :, -1]
            + f[1, :, -1]
            + f[3, :, -1]
            + 2 * (f[2, :, -1] + f[5, :, -1] + f[6, :, -1])
        ) / (1 + uy_wall)

    elif loc == "bottom":
        rho_wall = (
            f[0, :, 0]
            + f[1, :, 0]
            + f[3, :, 0]
            + 2 * (f[4, :, 0] + f[7, :, 0] + f[8, :, 0])
        ) / (1 - uy_wall)

    return _nebb(f, loc, ux_wall=ux_wall, uy_wall=uy_wall, rho_wall=rho_wall)


def nebb_pressure(f, loc: str, rho_wall=1):
    """Enforce given pressure (density) rho_wall at the specified boundary using the
    Non-Equilibrium Bounce-Back (or Zou/He) scheme.

    This function should be called after the streaming step. In this function,
    the velocity normal to the boundary is computed based on the known outgoing distribution functions.
    The velocity tangential to the boundary are taken from the fluid nodes next to the boundary.

    Args:
        f (jax.Array): Discrete distribution function (DDF)
        loc (str): The boundary where the pressure condition is enforced,
            can be 'left', 'right', 'top', or 'bottom'.
        rho_wall (scalar or jax.Array of shape NX or NY): The density at the wall.

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """

    if loc not in ["left", "right", "top", "bottom"]:
        raise ValueError(
            "Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'."
        )

    if loc == "left":
        ux_wall = (
            f[0, 0] + f[2, 0] + f[4, 0] + 2 * (f[3, 0] + f[6, 0] + f[7, 0])
        ) / rho_wall - 1
        rho_next = jnp.sum(f[:, 1], axis=0)
        uy_next = (
            jnp.sum(f[UP_DIRS, 1], axis=0) - jnp.sum(f[DOWN_DIRS, 1], axis=0)
        ) / rho_next
        uy_wall = uy_next

    elif loc == "right":
        ux_wall = (
            f[0, -1] + f[2, -1] + f[4, -1] + 2 * (f[1, -1] + f[5, -1] + f[8, -1])
        ) / rho_wall - 1
        rho_next = jnp.sum(f[:, -2], axis=0)
        uy_next = (
            jnp.sum(f[UP_DIRS, -2], axis=0) - jnp.sum(f[DOWN_DIRS, -2], axis=0)
        ) / rho_next
        uy_wall = uy_next

    elif loc == "top":
        uy_wall = (
            f[0, :, -1]
            + f[1, :, -1]
            + f[3, :, -1]
            + 2 * (f[2, :, -1] + f[5, :, -1] + f[6, :, -1])
        ) / rho_wall - 1
        rho_next = jnp.sum(f[:, :, -2], axis=0)
        ux_next = (
            jnp.sum(f[RIGHT_DIRS, :, -2], axis=0) - jnp.sum(f[LEFT_DIRS, :, -2], axis=0)
        ) / rho_next
        ux_wall = ux_next

    elif loc == "bottom":
        uy_wall = (
            f[0, :, 0]
            + f[1, :, 0]
            + f[3, :, 0]
            + 2 * (f[4, :, 0] + f[7, :, 0] + f[8, :, 0])
        ) / rho_wall - 1
        rho_next = jnp.sum(f[:, :, 1], axis=0)
        ux_next = (
            jnp.sum(f[RIGHT_DIRS, :, 1], axis=0) - jnp.sum(f[LEFT_DIRS, :, 1], axis=0)
        ) / rho_next
        ux_wall = ux_next

    return _nebb(f, loc, ux_wall=ux_wall, uy_wall=uy_wall, rho_wall=rho_wall)


def _nee(f, loc: str, u_wall, rho_wall):
    """Non-Equilibrium Extrapolation scheme.

    This is the core function for the nee_velocity and nee_pressure functions.

    It applies the Non-Equilibrium Extrapolation scheme to the distribution function
    at the specified boundary location, enforcing the given wall velocity and density.
    It modifies all the distribution functions at the boundary in such a way that
    ensures a zero-gradient condition for the non-equilibrium part of the
    distribution functions at the boundary.

    Args:
        f (jax.Array): Discrete distribution function (DDF)
        loc (str): The boundary where the velocity condition is enforced,
            can be 'left', 'right', 'top', or 'bottom'.
        u_wall (jax.Array of shape (2, NX) or (2, NY)): The velocity at the wall.
        rho_wall (scalar or jax.Array of shape NX or NY): The density at the wall.

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """

    feq_wall = get_equilibrium(rho_wall, u_wall)

    if loc == "left":
        rho_next, u_next = get_macroscopic(f[:, 1])
        fneq_next = f[:, 1] - get_equilibrium(rho_next, u_next)
        f = f.at[:, 0].set(feq_wall + fneq_next)

    elif loc == "right":
        rho_next, u_next = get_macroscopic(f[:, -2])
        fneq_next = f[:, -2] - get_equilibrium(rho_next, u_next)
        f = f.at[:, -1].set(feq_wall + fneq_next)

    elif loc == "top":
        rho_next, u_next = get_macroscopic(f[:, :, -2])
        fneq_next = f[:, :, -2] - get_equilibrium(rho_next, u_next)
        f = f.at[:, :, -1].set(feq_wall + fneq_next)

    elif loc == "bottom":
        rho_next, u_next = get_macroscopic(f[:, :, 1])
        fneq_next = f[:, :, 1] - get_equilibrium(rho_next, u_next)
        f = f.at[:, :, 0].set(feq_wall + fneq_next)

    return f


def nee_velocity(f, loc: str, ux_wall=0, uy_wall=0):
    """Enforce given velocity ux_wall, uy_wall at the specified boundary using the
    Non-Equilibrium Extrapolation scheme.

    This function should be called after the streaming step. In this function,
    the density at the wall is computed based on the known outgoing distribution functions.

    Args:
        f (jax.Array): Discrete distribution function (DDF)
        loc (str): The boundary where the velocity condition is enforced,
            can be 'left', 'right', 'top', or 'bottom'.
        ux_wall (scalar or jax.Array of shape NX): The x-component of velocity.
        uy_wall (scalar or jax.Array of shape NY): The y-component of velocity.

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF

    """

    if loc not in ["left", "right", "top", "bottom"]:
        raise ValueError(
            "Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'."
        )

    if loc == "left":
        rho_wall = (f[0, 0] + f[2, 0] + f[4, 0] + 2 * (f[3, 0] + f[6, 0] + f[7, 0])) / (
            1 - ux_wall
        )

    elif loc == "right":
        rho_wall = (
            f[0, -1] + f[2, -1] + f[4, -1] + 2 * (f[1, -1] + f[5, -1] + f[8, -1])
        ) / (1 + ux_wall)

    elif loc == "top":
        rho_wall = (
            f[0, :, -1]
            + f[1, :, -1]
            + f[3, :, -1]
            + 2 * (f[2, :, -1] + f[5, :, -1] + f[6, :, -1])
        ) / (1 + uy_wall)

    elif loc == "bottom":
        rho_wall = (
            f[0, :, 0]
            + f[1, :, 0]
            + f[3, :, 0]
            + 2 * (f[4, :, 0] + f[7, :, 0] + f[8, :, 0])
        ) / (1 - uy_wall)

    u_wall = jnp.zeros((2, rho_wall.shape[0]))
    u_wall = u_wall.at[0].set(ux_wall)
    u_wall = u_wall.at[1].set(uy_wall)

    return _nee(f, loc, u_wall=u_wall, rho_wall=rho_wall)


def nee_pressure(f, loc: str, rho_wall=1):
    """Enforce given pressure (density) rho_wall at the specified boundary using the
    Non-Equilibrium Extrapolation scheme.

    This function should be called after the streaming step. In this function,
    the velocity normal to the boundary is computed based on the known outgoing distribution functions.
    The velocity tangential to the boundary are taken from the fluid nodes next to the boundary.

    Args:
        f (jax.Array): Discrete distribution function (DDF)
        loc (str): The boundary where the pressure condition is enforced,
            can be 'left', 'right', 'top', or 'bottom'.
        rho_wall (scalar or jax.Array of shape NX or NY): The density at the wall.

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """

    if loc not in ["left", "right", "top", "bottom"]:
        raise ValueError(
            "Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'."
        )

    if loc == "left":
        rho_next = jnp.sum(f[:, 1], axis=0)
        uy_next = (
            jnp.sum(f[UP_DIRS, 1], axis=0) - jnp.sum(f[DOWN_DIRS, 1], axis=0)
        ) / rho_next
        ux_wall = (
            f[0, 0] + f[2, 0] + f[4, 0] + 2 * (f[3, 0] + f[6, 0] + f[7, 0])
        ) / rho_wall - 1
        u_wall = jnp.zeros((2, rho_next.shape[0]))
        u_wall = u_wall.at[0].set(ux_wall)
        u_wall = u_wall.at[1].set(uy_next)

    elif loc == "right":
        rho_next = jnp.sum(f[:, -2], axis=0)
        uy_next = (
            jnp.sum(f[UP_DIRS, -2], axis=0) - jnp.sum(f[DOWN_DIRS, -2], axis=0)
        ) / rho_next
        ux_wall = (
            f[0, -1] + f[2, -1] + f[4, -1] + 2 * (f[1, -1] + f[5, -1] + f[8, -1])
        ) / rho_wall - 1
        u_wall = jnp.zeros((2, rho_next.shape[0]))
        u_wall = u_wall.at[0].set(ux_wall)
        u_wall = u_wall.at[1].set(uy_next)

    elif loc == "top":
        rho_next = jnp.sum(f[:, :, -2], axis=0)
        ux_next = (
            jnp.sum(f[RIGHT_DIRS, :, -2], axis=0) - jnp.sum(f[LEFT_DIRS, :, -2], axis=0)
        ) / rho_next
        uy_wall = (
            f[0, :, -1]
            + f[1, :, -1]
            + f[3, :, -1]
            + 2 * (f[2, :, -1] + f[5, :, -1] + f[6, :, -1])
        ) / rho_wall - 1
        u_wall = jnp.zeros((2, rho_next.shape[0]))
        u_wall = u_wall.at[0].set(ux_next)
        u_wall = u_wall.at[1].set(uy_wall)

    elif loc == "bottom":
        rho_next = jnp.sum(f[:, :, 1], axis=0)
        ux_next = (
            jnp.sum(f[RIGHT_DIRS, :, 1], axis=0) - jnp.sum(f[LEFT_DIRS, :, 1], axis=0)
        ) / rho_next
        uy_wall = (
            f[0, :, 0]
            + f[1, :, 0]
            + f[3, :, 0]
            + 2 * (f[4, :, 0] + f[7, :, 0] + f[8, :, 0])
        ) / rho_wall - 1
        u_wall = jnp.zeros((2, rho_next.shape[0]))
        u_wall = u_wall.at[0].set(ux_next)
        u_wall = u_wall.at[1].set(uy_wall)

    rho_wall = jnp.ones_like(rho_next) * rho_wall

    return _nee(f, loc, u_wall=u_wall, rho_wall=rho_wall)
