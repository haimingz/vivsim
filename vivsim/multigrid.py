"""
This module contains functions for streaming particles across blocks with different resolutions.

Assumptions:
    - All blocks are rectangles. 
    - The outermost columns/rows of f are ghost layers to store temporary data.
    - The second outermost columns/rows of f are the real boundary of the block. 
    - The grid refinement ratio between two neighboring blocks is 2. 

Visualization of a block showing the ghost layers (G) and real boundaries (B).

    ---------------------------------   
    |   | G | G | G | G | G | G |   |
    ---------------------------------
    | G | B | B | B | B | B | B | G |
    ---------------------------------
    | G | B |   |   |   |   | B | G |
    ---------------------------------
    | G | B |   |   |   |   | B | G |
    ---------------------------------
    | G | B | B | B | B | B | B | G |
    ---------------------------------
    |   | G | G | G | G | G | G |   |
    --------------------------------- 

Coarse-to-fine transportation:
    The process is explosion (BS) -> propagate (AS) -> propagate (AS)
    - In the explosion step, the particles are transferred 
        from the coarse block to the ghost layer.
    - In the propagate step, the particles are transferred 
        from the ghost layer to the fine block.

Fine-to-coarse transportation:
    The process is accumulation (BS) -> accumulation (BS) -> coalescence (AS)
    - In the accumulation process, the particles are transferred
        from the fine block to the ghost layer.
    - In the coalescence step, the the particles are transferred
        from the ghost layer to the coarse block.
        
where BS stands for before streaming and AS stands for after streaming.
"""

import jax
import jax.numpy as jnp
from . import lbm

# lattice directions
RIGHT_DIRS = jnp.array([1, 5, 8])
LEFT_DIRS = jnp.array([3, 7, 6])
UP_DIRS = jnp.array([2, 5, 6])
DOWN_DIRS = jnp.array([4, 7, 8])


def explosion(f_coarse: jax.Array, f_fine: jax.Array, dir: str)-> jax.Array:
    """
    Transfer particles from the coarse block to the ghost layer. 
    
    This is the first step of coarse-to-fine transportation. It should be carried out
    BEFORE the streaming of the coarse block since it deals with the outgoing particles.
    
    Args:
        f_coarse (jax.Array): The distribution function of the coarse block.
        f_fine (jax.Array): The distribution function of the fine block.
        dir (str): The direction of the transportation, can be 'left', 'right', 'up', or 'down', 
                   For example, if dir='left', the particles are transporting from  
                   the block on the right to the block on the left.
    
    Returns:
        f_fine (jax.Array): The updated distribution functions of the fine block.
    """
    
    if dir == 'left':
        f_fine = f_fine.at[LEFT_DIRS, -1, 1:-1:2].set(f_coarse[LEFT_DIRS, 1, 1:-1])
        f_fine = f_fine.at[LEFT_DIRS, -1, 2:-1:2].set(f_coarse[LEFT_DIRS, 1, 1:-1])
    elif dir == 'right':
        f_fine = f_fine.at[RIGHT_DIRS, 0, 1:-1:2].set(f_coarse[RIGHT_DIRS, -2, 1:-1])
        f_fine = f_fine.at[RIGHT_DIRS, 0, 2:-1:2].set(f_coarse[RIGHT_DIRS, -2, 1:-1])
    elif dir == 'up':
        f_fine = f_fine.at[UP_DIRS, 1:-1:2, 0].set(f_coarse[UP_DIRS, 1:-1, -2])
        f_fine = f_fine.at[UP_DIRS, 2:-1:2, 0].set(f_coarse[UP_DIRS, 1:-1, -2])
    elif dir == 'down':
        f_fine = f_fine.at[DOWN_DIRS, 1:-1:2, -1].set(f_coarse[DOWN_DIRS, 1:-1, 1])
        f_fine = f_fine.at[DOWN_DIRS, 2:-1:2, -1].set(f_coarse[DOWN_DIRS, 1:-1, 1]) 
    return f_fine

def propagate(f: jax.Array, dir: str)-> jax.Array:
    """
    Transfer particles from the ghost layer into the block. 
    
    This is the second step of coarse-to-fine transportation. It should be carried out
    AFTER the streaming since it deals with the incoming particles.
    
    
    Args:
        f (jax.Array): The distribution function of the grid.
        dir (str): The direction of the transportation, can be 'left', 'right', 'up', or 'down', 
                   For example, if dir='left', the particles are transporting from  
                   the block on the right to the block on the left.
    
    Returns:
        f (jax.Array): The updated distribution functions of the grid.
    """
    
    if dir == 'left':
        f = f.at[6,-2,1:-1].set(jnp.roll(f[6,-1,1:-1],1,axis=0))
        f = f.at[7,-2,1:-1].set(jnp.roll(f[7,-1,1:-1],-1,axis=0))
        f = f.at[3,-2,1:-1].set(f[3,-1,1:-1])
    elif dir == 'right':
        f = f.at[5,1,1:-1].set(jnp.roll(f[5,0,1:-1],1,axis=0))
        f = f.at[8,1,1:-1].set(jnp.roll(f[8,0,1:-1],-1,axis=0))
        f = f.at[1,1,1:-1].set(f[1,0,1:-1])
    elif dir == 'up':
        f = f.at[5,1:-1,1].set(jnp.roll(f[5,1:-1,0],1,axis=0))
        f = f.at[6,1:-1,1].set(jnp.roll(f[6,1:-1,0],-1,axis=0))
        f = f.at[2,1:-1,1].set(f[2,1:-1,0])
    elif dir == 'down':
        f = f.at[8,1:-1,-2].set(jnp.roll(f[8,1:-1,-1],1,axis=0))
        f = f.at[7,1:-1,-2].set(jnp.roll(f[7,1:-1,-1],-1,axis=0))
        f = f.at[4,1:-1,-2].set(f[4,1:-1,-1])
    return f

def accumulate(f: jax.Array, dir: str)-> jax.Array:
    """
    Transfer particles from the fine block to the ghost layer.
    
    This is the first step of fine-to-coarse transportation. It should be carried out
    BEFORE the streaming of the fine block since it deals with the outgoing particles.
    
    Args:
        f (jax.Array): The distribution function of the fine block.
        dir (str): The direction of the transportation, can be 'left', 'right', 'up', or 'down', 
                   For example, if dir='left', the particles are transporting from  
                   the block on the right to the block on the left.
                   
    Returns:
        f (jax.Array): The updated distribution function of the fine block.
    """
    
    if dir == 'left':
        f = f.at[6, 0, 1:-1].add(jnp.roll(f[6, 1, 1:-1], 1, axis=0))
        f = f.at[7, 0, 1:-1].add(jnp.roll(f[7, 1, 1:-1], -1, axis=0))
        f = f.at[3, 0, 1:-1].add(f[3, 1, 1:-1])
    elif dir == 'right':
        f = f.at[5, -1, 1:-1].add(jnp.roll(f[5, -2, 1:-1], 1, axis=0))
        f = f.at[8, -1, 1:-1].add(jnp.roll(f[8, -2, 1:-1], -1, axis=0))
        f = f.at[1, -1, 1:-1].add(f[1, -2, 1:-1])
    elif dir == 'up':
        f = f.at[5, 1:-1, 0].add(jnp.roll(f[5, 1:-1, 1], 1, axis=0))
        f = f.at[6, 1:-1, 0].add(jnp.roll(f[6, 1:-1, 1], -1, axis=0))
        f = f.at[2, 1:-1, 0].add(f[2, 1:-1, 1])
    elif dir == 'down':
        f = f.at[8, 1:-1, -1].add(jnp.roll(f[8, 1:-1, -2], 1, axis=0))
        f = f.at[7, 1:-1, -1].add(jnp.roll(f[7, 1:-1, -2], -1, axis=0))
        f = f.at[4, 1:-1, -1].add(f[4, 1:-1, -2])
    return f 

def coalescence(f_fine: jax.Array, f_coarse: jax.Array, dir:str)-> jax.Array:
    """
    Transfer particles from the ghost layer to the coarse block.
    
    This is the second step of fine-to-coarse transportation. It should be carried out
    AFTER the streaming of the coarse block since it deals with the incoming particles.
    
    Args:
        f_fine (jax.Array): The distribution function of the fine block.
        f_coarse (jax.Array): The distribution function of the coarse block.
        dir (str): The direction of the transportation, can be 'left', 'right', 'up', or 'down', 
                   For example, if dir='left', the particles are transporting from  
                   the block on the right to the block on the left.
                   
    Returns:
        f_coarse (jax.Array): The updated distribution function of the coarse block.
    """
    
    if dir == 'left':
        return f_coarse.at[LEFT_DIRS, -2, 1:-1].set((f_fine[LEFT_DIRS, 0, 1:-1:2] + f_fine[LEFT_DIRS, 0, 2:-1:2]) / 4)
    elif dir == 'right':
        return f_coarse.at[RIGHT_DIRS, 1, 1:-1].set((f_fine[RIGHT_DIRS, -1, 1:-1:2] + f_fine[RIGHT_DIRS, -1, 2:-1:2]) / 4)
    elif dir == 'up':
        return f_coarse.at[UP_DIRS, 1:-1, 1].set((f_fine[UP_DIRS, 1:-1:2, -1] + f_fine[UP_DIRS, 2:-1:2, -1]) / 4)
    elif dir == 'down':
        return f_coarse.at[DOWN_DIRS, 1:-1, -2].set((f_fine[DOWN_DIRS, 1:-1:2, 0] + f_fine[DOWN_DIRS, 2:-1:2, 0]) / 4)


def clear_ghost(f: jax.Array, location:str)-> jax.Array:
    """
    Clear the ghost layer of the coarse block.
    
    Args:
        f_fine (jax.Array): The distribution function of the coarse block.
        location (str): The location of the ghost layer. 
            It can be 'left', 'right', 'top', or 'bottom'.
        
    Returns:
        f_fine (jax.Array): The updated distribution function of the coarse block.
    """
    
    if location == 'left' or location == 'all':
        return f.at[:, 0].set(0)
    elif location == 'right' or location == 'all':
        return f.at[:, -1].set(0)
    elif location == 'top' or location == 'all':
        return f.at[:, :, -1].set(0)
    elif location == 'bottom' or location == 'all':
        return f.at[:, :, 0].set(0)
     
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
    
    Note that the size of f is two column/row larger than the size of other variables because
    the outermost column/row of f is reserved for the ghost layer.
    
    Args:
        width (int): The width of the block (in lattice unit).
        height (int): The height of the block (in lattice unit).
        level (int): The level of refinement. 
            Default is 0, can be negative number. 
            Greater level means finer grid.
            Smaller level means coarser grid.
    """
  
    nx = int(width * 2 ** level)
    ny = int(height * 2 ** level)
    
    # fluid variables
    f = jnp.zeros((9, nx + 2, ny + 2), dtype=jnp.float32)
    feq = jnp.zeros((9, nx, ny), dtype=jnp.float32) 
    rho = jnp.ones((nx, ny), dtype=jnp.float32) 
    u = jnp.zeros((2, nx, ny), dtype=jnp.float32)
    
    return f, feq, rho, u

def streaming(f):
    return f.at[:, 1:-1, 1:-1].set(lbm.streaming(f[:, 1:-1, 1:-1]))