"""
This module implements the equilibrium boundary condition for the Lattice Boltzmann Method (LBM)
in a 2D D2Q9 lattice.

The equilibrium boundary condition assigns the distribution functions at the boundary to their
equilibrium values based on the specified density and velocity at the boundary. This method is
less accurate than the non-equilibrium bounce-back (NEBB) or non-equilibrium extrapolation (NEE)
schemes but is far more stable and easier to implement. It also supresses sound wave reflections
at the boundary.

When body forces are present, pass the corrected wall velocity obtained from
``get_corrected_wall_velocity`` (from this package) before calling these functions.
"""



import jax.numpy as jnp
from ..basic import get_equilibrium


def boundary_equilibrium(f, loc:str, rho_wall=1, ux_wall=0, uy_wall=0):
    """Enforce given density rho_wall and velocity ux_wall, uy_wall
    at the specified boundary using the equilibrium scheme.
    
    This is a simple boundary condition that sets the distribution functions
    at the boundary to their equilibrium values based on the specified
    density and velocity. This method is less accurate than the NEBB or NEE
    schemes but more stable. 
    
    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        loc (str): The boundary where the condition is enforced, 
            can be 'left', 'right', 'top', or 'bottom'.
        rho_wall (scalar or jax.Array of shape NX or NY): The density at the wall.
        ux_wall (scalar or jax.Array of shape NX or NY): The x-component of velocity.
        uy_wall (scalar or jax.Array of shape NX or NY): The y-component of velocity.
        
    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """
    
    # Determine boundary size and convert scalars to arrays if needed
    size = f.shape[2] if loc in ['left', 'right'] else f.shape[1]
    
    if jnp.isscalar(rho_wall):
        rho_wall = jnp.full(size, rho_wall)
    if jnp.isscalar(ux_wall):
        ux_wall = jnp.full(size, ux_wall)
    if jnp.isscalar(uy_wall):
        uy_wall = jnp.full(size, uy_wall)
    
    u_wall = jnp.array([ux_wall, uy_wall])
    feq_wall = get_equilibrium(rho_wall, u_wall)
    
    # Apply boundary condition
    if loc == "left":
        f = f.at[:, 0].set(feq_wall)
    elif loc == "right":
        f = f.at[:, -1].set(feq_wall)
    elif loc == "top":
        f = f.at[:, :, -1].set(feq_wall)
    elif loc == "bottom":
        f = f.at[:, :, 0].set(feq_wall)
    return f