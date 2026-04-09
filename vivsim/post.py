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
@jax.jit
def calculate_q_criterion(u):
    """
    Calculate the Q-criterion for vortex identification in a 2D velocity field.

    Q = 0.5 * (||Omega||^2 - ||S||^2)
    where Omega is the anti-symmetric part of the velocity gradient tensor (vorticity tensor)
    and S is the symmetric part (rate-of-strain tensor).

    Args:
        u (ndarray): The velocity field of the fluid of shape (2, NX, NY).

    Returns:
        ndarray: The Q-criterion field.
    """
    ux_x = jnp.gradient(u[0], axis=0)
    ux_y = jnp.gradient(u[0], axis=1)
    uy_x = jnp.gradient(u[1], axis=0)
    uy_y = jnp.gradient(u[1], axis=1)

    # In 2D, Q can be simplified to:
    # Q = -0.5 * (ux_x^2 + uy_y^2 + 2*ux_y*uy_x)
    # However, for incompressible flow, ux_x + uy_y = 0, so ux_x^2 + uy_y^2 = 2*ux_x^2 = -2*ux_x*uy_y
    # Thus Q = ux_x*uy_y - ux_y*uy_x (determinant of velocity gradient tensor)
    return ux_x * uy_y - ux_y * uy_x

@jax.jit
def calculate_kinetic_energy(u, rho=1.0):
    """
    Calculate the total kinetic energy of the fluid.

    Args:
        u (ndarray): The velocity field of the fluid.
        rho (ndarray or float): The density field of the fluid.

    Returns:
        float: The total kinetic energy.
    """
    return 0.5 * jnp.sum(rho * (u[0]**2 + u[1]**2))

@jax.jit
def calculate_enstrophy(u):
    """
    Calculate the enstrophy of the fluid.

    Args:
        u (ndarray): The velocity field of the fluid.

    Returns:
        float: The enstrophy.
    """
    vorticity = calculate_vorticity(u)
    return 0.5 * jnp.sum(vorticity**2)

@jax.jit
def calculate_pressure(rho, cs_lat=1/jnp.sqrt(3)):
    """
    Calculate pressure from density using the ideal gas equation of state in LBM.

    Args:
        rho (ndarray): The density field of the fluid.
        cs_lat (float): The lattice speed of sound.

    Returns:
        ndarray: The pressure field.
    """
    return rho * (cs_lat**2)

@jax.jit
def calculate_force_coefficient(force, l, u0, rho0=1.0):
    """
    Calculate a force coefficient (e.g., drag or lift coefficient).

    Args:
        force (float): The force component (e.g., drag or lift).
        l (float): Characteristic length (e.g., cylinder diameter).
        u0 (float): Characteristic velocity.
        rho0 (float): Reference density.

    Returns:
        float: Force coefficient.
    """
    return 2.0 * force / (rho0 * l * u0**2)

@jax.jit
def calculate_strouhal_number(frequency, l, u0):
    """
    Calculate the Strouhal number.

    Args:
        frequency (float): Vortex shedding frequency.
        l (float): Characteristic length (e.g., cylinder diameter).
        u0 (float): Characteristic velocity.

    Returns:
        float: Strouhal number.
    """
    return frequency * l / u0
