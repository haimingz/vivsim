"""
This module contains functions for transporting particles across device boundaries

The computational domain should be divided evenly with straight boundaries among devices
along the x and y directions. The functions in this module ensure that particles are 
transported across device boundaries to maintain continuity of particle transport.
"""

import jax
import jax.numpy as jnp


# lattice directions
RIGHT_DIRS = jnp.array([1, 5, 8])
LEFT_DIRS = jnp.array([3, 7, 6])
UP_DIRS = jnp.array([2, 5, 6])
DOWN_DIRS = jnp.array([4, 7, 8])


def stream_cross_devices(f: jax.Array, dir: str, device_axis: str , n_devices: int)  -> jax.Array:
    """Stream particles across device boundaries. The computational domain is divided evenly
    into a series of subdomains along the x or y axis, each corresponding to a single device.
    
    Args:
        f (jax.Array): Distribution functions
        dir (str): Direction of streaming ('x' or 'y')
        n_devices (int): Number of devices in the direction
        device_axis (int): the axis name of the devices, assigned when creating the Mesh
            of devices
    
    Returns:
        jax.Array: Distribution functions after streaming
    """

    device_pairs = [(i, (i + 1) % n_devices) for i in range(n_devices)]
    device_pairs_reverse = [((i + 1) % n_devices, i) for i in range(n_devices)]
    
    if dir == 'y':
        f = f.at[UP_DIRS, :, 0].set(jax.lax.ppermute(f[UP_DIRS, :, 0], device_axis, device_pairs))
        f = f.at[DOWN_DIRS, :, -1].set(jax.lax.ppermute(f[DOWN_DIRS, :, -1], device_axis, device_pairs_reverse))
    
    elif dir == 'x':
        f = f.at[RIGHT_DIRS, 0].set(jax.lax.ppermute(f[RIGHT_DIRS, 0], device_axis, device_pairs_reverse))
        f = f.at[LEFT_DIRS, -1].set(jax.lax.ppermute(f[LEFT_DIRS, -1], device_axis, device_pairs))
    
    return f

