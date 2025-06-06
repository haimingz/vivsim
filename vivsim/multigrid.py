"""
This module contains functions for grid refinement. 

This module adopted a buffer layer approach, which adds an overlapping region of cells at each boundary
between blocks of different levels of refinement. By convention, only a level 2 refinement grid is allowed
for two neighboring blocks. 

There are two types of data transfer between blocks of different levels of refinement, i.e.,
- fine_to_coarse: transfer data from a fine grid to a coarser grid, and 
- coarse_to_fine: transfer data from a coarser grid to a fine grid.
These operations can be seen as special boundary condition treatments; they can be called after streaming 
to fill up the missing incoming DDFs. 
"""


import jax.numpy as jnp

from .lbm import LEFT_DIRS, RIGHT_DIRS, UP_DIRS, DOWN_DIRS


def init_grid(width, height, level=0, buffer_x=0, buffer_y=0):
    """
    Generate the fluid variables for a block of fluid on a grid with a given level of refinement.
    
    The shape of the data is determined by the width and height of the block and the level of refinement.
    Buffer layer, an overlapping region of cells, can be added at each boundary for data transfer 
    between blocks of different levels of refinement. By convention, the buffer layer is added to the 
    coarser grid. This means that if the neighboring block is a finer grid, the buffer layer is added to
    the current block, as vice versa. 
    
    Args:
        width (int): The width of the block (in lattice unit).
        height (int): The height of the block (in lattice unit).
        level (int): The level of refinement. 
            Default is 0, can be negative number. 
            Greater level means finer grid.
            Smaller level means coarser grid.
        buffer_x (int): The number of buffer cells in the x-direction.
            Default is 0.
        buffer_y (int): The number of buffer cells in the y-direction.
            Default is 0.
    
    Returns:
        f (jnp.ndarray): The DDF of the fluid variables.
    """

    nx = int(width * 2 ** level) + buffer_x
    ny = int(height * 2 ** level) + buffer_y
        
    # DDF
    f = jnp.zeros((9, nx, ny), dtype=jnp.float32)
    rho = jnp.ones((nx, ny), dtype=jnp.float32)
    u = jnp.zeros((2, nx, ny), dtype=jnp.float32)
    
    return f, rho, u


def fine_to_coarse(f_fine, f_coarse, dir):
    """Transfer DDFs from a fine grid to a coarser grid along a given direction.
    
    Args:
        f_fine (jnp.ndarray): The DDFs of the fine grid.
        f_coarse (jnp.ndarray): The DDFs of the coarse grid.
        dir (str): The direction of transfer. 
            'left', 'right', 'up', 'down'.
            
    Returns:
        f_coarse (jnp.ndarray): The updated DDFs of the coarse grid.
    """
    
    if dir ==  'left':
        f_coarse = f_coarse.at[LEFT_DIRS, -1].set(
            0.25 * (f_fine[LEFT_DIRS, 0, 0::2] 
                  + f_fine[LEFT_DIRS, 0, 1::2] 
                  + f_fine[LEFT_DIRS, 1, 0::2] 
                  + f_fine[LEFT_DIRS, 1, 1::2]))
    
    if dir == 'right':
        f_coarse = f_coarse.at[RIGHT_DIRS, 0].set(
            0.25 * (f_fine[RIGHT_DIRS, -1, 0::2] 
                  + f_fine[RIGHT_DIRS, -1, 1::2] 
                  + f_fine[RIGHT_DIRS, -2, 0::2] 
                  + f_fine[RIGHT_DIRS, -2, 1::2]))
        
    if dir == 'up':
        f_coarse = f_coarse.at[UP_DIRS, :, 0].set(
            0.25 * (f_fine[UP_DIRS, 0::2, -1] 
                  + f_fine[UP_DIRS, 1::2, -1] 
                  + f_fine[UP_DIRS, 0::2, -2] 
                  + f_fine[UP_DIRS, 1::2, -2]))
    if dir == 'down':
        f_coarse = f_coarse.at[DOWN_DIRS, :, -1].set(
            0.25 * (f_fine[DOWN_DIRS, 0::2, 0] 
                  + f_fine[DOWN_DIRS, 1::2, 0] 
                  + f_fine[DOWN_DIRS, 0::2, 1] 
                  + f_fine[DOWN_DIRS, 1::2, 1]))
    
    return f_coarse
        
def coarse_to_fine(f_coarse, f_fine, dir):
    """Transfer DDFs from a coarser grid to a fine grid along a given direction.
    
    Args:
        f_coarse (jnp.ndarray): The DDFs of the coarse grid.
        f_fine (jnp.ndarray): The DDFs of the fine grid.
        dir (str): The direction of transfer. 
            'left', 'right', 'up', 'down'.
    
    Returns:
        f_fine (jnp.ndarray): The updated DDFs of the fine grid.
    """
    
    if dir == 'left':
        f_fine = f_fine.at[LEFT_DIRS, -1, 0::2].set(f_coarse[LEFT_DIRS, 0])
        f_fine = f_fine.at[LEFT_DIRS, -1, 1::2].set(f_coarse[LEFT_DIRS, 0])
    
    if dir == 'right':
        f_fine = f_fine.at[RIGHT_DIRS, 0, 0::2].set(f_coarse[RIGHT_DIRS, -1])
        f_fine = f_fine.at[RIGHT_DIRS, 0, 1::2].set(f_coarse[RIGHT_DIRS, -1])
    
    if dir == 'up':
        f_fine = f_fine.at[UP_DIRS, 0::2, 0].set(f_coarse[UP_DIRS, :, -1])
        f_fine = f_fine.at[UP_DIRS, 1::2, 0].set(f_coarse[UP_DIRS, :, -1])
        
    if dir == 'down':
        f_fine = f_fine.at[DOWN_DIRS, 0::2, -1].set(f_coarse[DOWN_DIRS, :, 0])
        f_fine = f_fine.at[DOWN_DIRS, 1::2, -1].set(f_coarse[DOWN_DIRS, :, 0])
        
    return f_fine


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


def coord_to_indices(x, y, grid_start_x, grid_start_y, level=0):
    """ 
    Convert global coordinates to local indices within a refined grid level.
    Given a global (x, y) coordinate, and the starting coordinates of a grid block,
    this function computes the corresponding local indices at a specified refinement level.
    The refinement level determines the fineness of the grid; higher levels correspond
    to finer grids, while lower levels represent coarser grids.
        
    Args:
        grid_start_x (int): The x-coordinate of the grid block's origin.
        grid_start_y (int): The y-coordinate of the grid block's origin.
        level (int, optional): The refinement level. Defaults to 0.
            Positive levels indicate finer grids (higher resolution), while negative
            levels indicate coarser grids (lower resolution).
    Returns:
        tuple of int: A tuple containing the local x and y indices (local_x, local_y).
    """
    
    local_x = int((x - grid_start_x) * 2 ** level)
    local_y = int((y - grid_start_y) * 2 ** level)
    
    return local_x, local_y