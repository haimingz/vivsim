import jax 
import jax.numpy as jnp


@jax.jit
def calculate_curl(u):
    ux_y = jnp.gradient(u[0], axis=1)
    uy_x = jnp.gradient(u[1], axis=0)
    curl = ux_y - uy_x
    return curl

@jax.jit
def calculate_vorticity(u):
    """
    Calculate the vorticity of a velocity field.
    
    Args:
        u (ndarray): The velocity field of the fluid.
        
    Returns:
        ndarray: The vorticity of the fluid.    
    """
    return calculate_curl(u)

@jax.jit
def calculate_vorticity_dimensionless(u, l, u0):
    """
    Calculate the dimensionless vorticity of a velocity field.
    
    Args:
        u (ndarray): The velocity field of the fluid.
        l (float): The characteristic length.
        u0 (float): The flow  velocity.
    
    Returns:
        ndarray: The dimensionless vorticity of the fluid.
    """
    curl = calculate_curl(u)
    return curl * l / u0


@jax.jit 
def calculate_velocity_magnitude(u):
    """
    Calculate the magnitude of the velocity field.
    
    Args:
        u (ndarray): The velocity field of the fluid.
        
    Returns:
        ndarray: The magnitude of the velocity field.
    """
    return jnp.sqrt(u[0]**2 + u[1]**2)