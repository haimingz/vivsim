"""
This module contains functions for transporting particles across subdomains with 
different resolutions.
"""

import jax
import jax.numpy as jnp

# lattice directions
RIGHT_DIRS = jnp.array([1, 5, 8])
LEFT_DIRS = jnp.array([3, 7, 6])
UP_DIRS = jnp.array([2, 5, 6])
DOWN_DIRS = jnp.array([4, 7, 8])


def explosion(f_fine: jax.Array, f_coarse: jax.Array, dir: str)-> tuple[jax.Array, jax.Array]:
    """
    Transfer particles from the coarse grid to the fine grid. Note that the outermost layer
    of the coarse grid is used as a 'ghost' layer to store information from the fine grid.
    Therefore particles from the second outermost layer of the coarse grid is transferred
    to the fine grid.
    
    Args:
        f_fine (jax.Array): The distribution function on the fine grid.
        f_coarse (jax.Array): The distribution function on the coarse grid.
        dir (str): The direction of the explosion. It can be 'left', 'right', 'up', or 'down', 
                   For example, if dir='left', the coarse grid should be on the right, 
                   transporting particles to the fine grid on the left. 
    
    Returns:
        f_fine (jax.Array): The updated distribution functions on the fine grid.
        f_coarse (jax.Array): The updated distribution functions on the coarse grid.
    """
    
    if dir == 'left':
        f_fine = f_fine.at[LEFT_DIRS, -1, 0::2].set(f_coarse[LEFT_DIRS, 1])
        f_fine = f_fine.at[LEFT_DIRS, -1, 1::2].set(f_coarse[LEFT_DIRS, 1])
    elif dir == 'right':
        f_fine = f_fine.at[RIGHT_DIRS, 0, 0::2].set(f_coarse[RIGHT_DIRS, -2])
        f_fine = f_fine.at[RIGHT_DIRS, 0, 1::2].set(f_coarse[RIGHT_DIRS, -2])
    elif dir == 'up':
        f_fine = f_fine.at[UP_DIRS, 0::2, 0].set(f_coarse[UP_DIRS, :, -2])
        f_fine = f_fine.at[UP_DIRS, 1::2, 0].set(f_coarse[UP_DIRS, :, -2])
    elif dir == 'down':
        f_fine = f_fine.at[DOWN_DIRS, 0::2, -1].set(f_coarse[DOWN_DIRS, :, 1])
        f_fine = f_fine.at[DOWN_DIRS, 1::2, -1].set(f_coarse[DOWN_DIRS, :, 1]) 
    return f_fine, f_coarse

def accumulate(f_fine: jax.Array, f_coarse: jax.Array, dir: str)-> tuple[jax.Array, jax.Array]:
    """
    This is the first step of transferring particles from the fine grid to the coarse grid. 
    In this step, particles from the fine grid are accumulated in the ghost layer of the coarse grid,
    which is the outermost layer of the coarse grid.
    
    Args:
        f_fine (jax.Array): The distribution function on the fine grid.
        f_coarse (jax.Array): The distribution function on the coarse grid.
        dir (str): The direction of the accumulation. It can be 'left', 'right', 'up', or 'down', 
                   For example, if dir='left', the fine grid should be on the left,
                   transporting particles to the coarse grid on the left.
                   
    Returns:
        f_fine (jax.Array): The updated distribution functions on the fine grid.
        f_coarse (jax.Array): The updated distribution functions on the coarse grid.
    """
    
    if dir == 'left':
        return f_coarse.at[LEFT_DIRS, -1].add(f_fine[LEFT_DIRS, 0, 0::2] + f_fine[LEFT_DIRS, 0, 1::2])
    elif dir == 'right':
        return f_coarse.at[RIGHT_DIRS, 0].add(f_fine[RIGHT_DIRS,-1, 0::2] + f_fine[RIGHT_DIRS,-1, 1::2])
    elif dir == 'up':
        return f_coarse.at[UP_DIRS, :, 0].add(f_fine[UP_DIRS, 0::2, -1] + f_fine[UP_DIRS, 1::2, -1])
    elif dir == 'down':
        return f_coarse.at[DOWN_DIRS, :, -1].add(f_fine[DOWN_DIRS, 0::2, 0] + f_fine[DOWN_DIRS, 1::2, 0])


def coalescence(f_coarse: jax.Array, dir:str)-> jax.Array:
    """
    This is the second step of transferring particles from the fine grid to the coarse grid.
    In this step, particles accumulated in the ghost layer of the coarse grid are coalesced
    into the real coarse grid. 
    
    Args:
        f_coarse (jax.Array): The distribution function on the coarse grid.
        dir (str): The direction of the coalescence. It can be 'left', 'right', 'up', or 'down', 
                   For example, if dir='left', the fine grid should be on the left,
                   transporting particles to the coarse grid on the left.
                   
    Returns:
        f_coarse (jax.Array): The updated distribution functions on the coarse grid.
    """
    
    if dir == 'left':
        return f_coarse.at[LEFT_DIRS, -2].set(f_coarse[LEFT_DIRS, -1]/4)
    elif dir == 'right':
        return f_coarse.at[RIGHT_DIRS, 1].set(f_coarse[RIGHT_DIRS, 0]/4)
    elif dir == 'up':
        return f_coarse.at[UP_DIRS, :, -2].set(f_coarse[UP_DIRS, :, -1]/4)
    elif dir == 'down':
        return f_coarse.at[DOWN_DIRS, :, 1].set(f_coarse[DOWN_DIRS, :, 0]/4)


def clear_ghost(f_coarse: jax.Array, location:str)-> jax.Array:
    """
    Clear the ghost layer of the coarse grid.
    
    Args:
        f_coarse (jax.Array): The distribution function on the coarse grid.
        location (str): The location of the ghost layer. It can be 'left', 'right', 'top', or 'bottom'.
        
    Returns:
        f_coarse (jax.Array): The updated distribution functions on the coarse grid.
    """
    
    if location == 'left':
        return f_coarse.at[:, 0].set(0)
    elif location == 'right':
        return f_coarse.at[:, -1].set(0)
    elif location == 'top':
        return f_coarse.at[:, :, 0].set(0)
    elif location == 'bottom':
        return f_coarse.at[:, :, -1].set(0)
    
    
def get_omega(nu, level=0):
    """
    Compute the relaxation parameter omega for different levels of refinement.
    
    Args:
        nu (float): The kinematic viscosity (in lattice unit).
        level (int): The level of refinement. 
            Default is 0. 
            Greater level means finer grid.
            Negative level means coarser grid.
    """ 
    omega_l0 = 1 / (3 * nu + 0.5)
    omega = 2 * omega_l0 / ( 2 ** (level + 1) + (1 - 2 ** level) * omega_l0)
    
    return omega


def generate_block_data(width, height, level=0):
    """
    Generate the fluid variables for a block of fluid on a grid with a given level of refinement.
    
    Args:
        width (int): The width of the block (in lattice unit).
        height (int): The height of the block (in lattice unit).
        level (int): The level of refinement. 
            Default is 0. 
            Greater level means finer grid.
            Negative level means coarser grid.
    """
  
    nx = int(width * 2 ** level)
    ny = int(height * 2 ** level)
    
    # fluid variables
    f = jnp.zeros((9, nx, ny), dtype=jnp.float32)
    feq = jnp.zeros((9, nx, ny), dtype=jnp.float32) 
    rho = jnp.ones((nx, ny), dtype=jnp.float32) 
    u = jnp.zeros((2, nx, ny), dtype=jnp.float32)
    
    return f, feq, rho, u