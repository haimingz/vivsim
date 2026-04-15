import jax
import jax.numpy as jnp
from ..basic import VELOCITIES, WEIGHTS

def get_shan_chen_force(rho: jax.Array, g: float, psi_func=None) -> jax.Array:
    """Calculate the Shan-Chen multiphase interaction force.

    The Shan-Chen multiphase model uses a pseudopotential interaction force
    to simulate multiphase flow. This force is responsible for phase separation
    and surface tension.

    Args:
        rho (jax.Array of shape (NX, NY)): Macroscopic density field.
        g (float): Interaction strength parameter (G). Typically a negative value
            for attraction between particles of the same phase.
        psi_func (Callable): Function to calculate pseudopotential from density.
            If None, uses the default psi(rho) = 1 - exp(-rho).

    Returns:
        force (jax.Array of shape (2, NX, NY)): The calculated Shan-Chen force field.
    """
    if psi_func is None:
        psi = 1.0 - jnp.exp(-rho)
    else:
        psi = psi_func(rho)

    def accumulate_force(weight, velocity):
        # We need psi(x + e_i), which means we shift psi by -e_i to align it with x
        psi_neighbor = jnp.roll(psi, shift=(-velocity[0], -velocity[1]), axis=(0, 1))
        return weight * psi_neighbor * velocity.reshape((2, 1, 1))

    # Calculate force contributions from all directions
    forces_all_dirs = jax.vmap(accumulate_force)(WEIGHTS, VELOCITIES)

    # Sum up over all directions
    force = jnp.sum(forces_all_dirs, axis=0)

    # Multiply by -g * psi(x)
    force = -g * psi * force

    return force
