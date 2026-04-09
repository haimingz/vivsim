import jax
import jax.numpy as jnp

def convert_velocity_to_lattice(u_phys, dx, dt):
    """
    Convert physical velocity to lattice velocity.

    Args:
        u_phys: Physical velocity.
        dx: Grid spacing (physical length per lattice unit).
        dt: Time step (physical time per lattice unit).

    Returns:
        Lattice velocity.
    """
    return u_phys * dt / dx

def convert_velocity_to_physical(u_lat, dx, dt):
    """
    Convert lattice velocity to physical velocity.

    Args:
        u_lat: Lattice velocity.
        dx: Grid spacing (physical length per lattice unit).
        dt: Time step (physical time per lattice unit).

    Returns:
        Physical velocity.
    """
    return u_lat * dx / dt

def convert_viscosity_to_lattice(nu_phys, dx, dt):
    """
    Convert physical kinematic viscosity to lattice viscosity.

    Args:
        nu_phys: Physical kinematic viscosity.
        dx: Grid spacing.
        dt: Time step.

    Returns:
        Lattice viscosity.
    """
    return nu_phys * dt / (dx ** 2)

def convert_viscosity_to_physical(nu_lat, dx, dt):
    """
    Convert lattice viscosity to physical kinematic viscosity.

    Args:
        nu_lat: Lattice viscosity.
        dx: Grid spacing.
        dt: Time step.

    Returns:
        Physical kinematic viscosity.
    """
    return nu_lat * (dx ** 2) / dt

def calculate_reynolds_number(u, l, nu):
    """
    Calculate the Reynolds number.

    Args:
        u: Characteristic velocity.
        l: Characteristic length.
        nu: Kinematic viscosity.

    Returns:
        Reynolds number.
    """
    return u * l / nu

def calculate_mach_number(u_lat, cs_lat=1/jnp.sqrt(3)):
    """
    Calculate the Mach number in lattice units.

    Args:
        u_lat: Lattice velocity magnitude.
        cs_lat: Lattice speed of sound (default is 1/sqrt(3) for D2Q9).

    Returns:
        Mach number.
    """
    return u_lat / cs_lat
