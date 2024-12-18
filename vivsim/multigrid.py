import jax
import jax.numpy as jnp

# lattice directions
right_dirs = jnp.array([1, 5, 8])
left_dirs = jnp.array([3, 7, 6])
top_dirs = jnp.array([2, 5, 6])
btm_dirs = jnp.array([4, 7, 8])


def explosion(f_fine: jax.Array, f_coarse: jax.Array, dir: str)-> tuple[jax.Array, jax.Array]:
    """
    Transfer particles from the coarse grid to the fine grid.
    
    Args:
        f_fine (jax.Array): The distribution function on the fine grid.
        f_coarse (jax.Array): The distribution function on the coarse grid.
        dir (str): The direction of the boundary. It can be 'left', 'right', 'top', or 'btm', 
                   depending on the location of the fine and coarse grids. For example, if dir='left',
                   the coarse grid should be on the right while the fine grid should be on the left.
    
    Returns:
        f_fine (jax.Array): The updated distribution functions on the fine grid.
        f_coarse (jax.Array): The updated distribution functions on the coarse grid.
    """
    
    if dir == 'left':
        f_fine = f_fine.at[left_dirs, -1, 0::2].set(f_coarse[left_dirs, 1])
        f_fine = f_fine.at[left_dirs, -1, 1::2].set(f_coarse[left_dirs, 1])
    elif dir == 'right':
        f_fine = f_fine.at[right_dirs, 0, 0::2].set(f_coarse[right_dirs, -2])
        f_fine = f_fine.at[right_dirs, 0, 1::2].set(f_coarse[right_dirs, -2])
    elif dir == 'top':
        f_fine = f_fine.at[top_dirs, 0::2, 0].set(f_coarse[top_dirs, :, -2])
        f_fine = f_fine.at[top_dirs, 1::2, 0].set(f_coarse[top_dirs, :, -2])
    elif dir == 'btm':
        f_fine = f_fine.at[btm_dirs, 0::2, -1].set(f_coarse[btm_dirs, :, 1])
        f_fine = f_fine.at[btm_dirs, 1::2, -1].set(f_coarse[btm_dirs, :, 1]) 
    return f_fine, f_coarse

def accumulate(f_fine: jax.Array, f_coarse: jax.Array, dir: str)-> tuple[jax.Array, jax.Array]:    
    
    if dir == 'left':
        f_coarse = f_coarse.at[left_dirs, -1].add(f_fine[left_dirs, 0, 0::2] + f_fine[left_dirs, 0, 1::2])
    elif dir == 'right':
        f_coarse = f_coarse.at[right_dirs, 0].add(f_fine[right_dirs,-1, 0::2] + f_fine[right_dirs,-1, 1::2])
    elif dir == 'top':
        f_coarse = f_coarse.at[top_dirs, :, 0].add(f_fine[top_dirs, 0::2, -1] + f_fine[top_dirs, 1::2, -1])
    elif dir == 'btm':
        f_coarse = f_coarse.at[btm_dirs, :, -1].add(f_fine[btm_dirs, 0::2, 0] + f_fine[btm_dirs, 1::2, 0])
    return  f_coarse


def coalescence(f_coarse: jax.Array, dir:str)-> jax.Array:
    
    if dir == 'left':
        f_coarse = f_coarse.at[left_dirs, -2].set(f_coarse[left_dirs, -1]/4)
    elif dir == 'right':
        f_coarse = f_coarse.at[right_dirs, 1].set(f_coarse[right_dirs, 0]/4)
    elif dir == 'top':
        f_coarse = f_coarse.at[top_dirs, :, -2].set(f_coarse[top_dirs, :, -1]/4)
    elif dir == 'btm':
        f_coarse = f_coarse.at[btm_dirs, :, 1].set(f_coarse[btm_dirs, :, 0]/4)
    return f_coarse

def clear_ghost(f_coarse: jax.Array, location:str)-> jax.Array:
    
    if location == 'left':
        f_coarse = f_coarse.at[:, 0].set(0)
    elif location == 'right':
        f_coarse = f_coarse.at[:, -1].set(0)
    elif location == 'top':
        f_coarse = f_coarse.at[:, :, 0].set(0)
    elif location == 'btm':
        f_coarse = f_coarse.at[:, :, -1].set(0)
    return f_coarse