import jax
import jax.numpy as jnp

from .basic import VELOCITIES, WEIGHTS, OPP_DIRS

TYPE_FLUID = 1
TYPE_INTERFACE = 0
TYPE_GAS = -1

def get_gas_equilibrium(rho_gas, u):
    ndim = 2
    uc = jnp.sum(u[None, ...] * VELOCITIES.reshape((9, 2, 1, 1)), axis=1)
    w_shape = (9, 1, 1)
    feq = (rho_gas * WEIGHTS.reshape(w_shape) *
          (1 + 3 * uc + 4.5 * uc ** 2 - 1.5 * jnp.sum(u ** 2, axis=0)))
    return feq

def reconstruct_interface_distributions(f_post, f_streamed, cell_type, rho_gas, u):
    def shift_fn(shift):
        return jnp.roll(cell_type, shift=shift, axis=(0, 1))

    neighbor_type_in = jax.vmap(shift_fn)(VELOCITIES)

    came_from_gas = neighbor_type_in == TYPE_GAS

    feq_gas = get_gas_equilibrium(rho_gas, u)

    f_post_opp = f_post[OPP_DIRS]
    feq_gas_opp = feq_gas[OPP_DIRS]

    f_reconstructed = feq_gas + feq_gas_opp - f_post_opp

    mask = (cell_type[None, ...] == TYPE_INTERFACE) & came_from_gas
    f_new = jnp.where(mask, f_reconstructed, f_streamed)
    return f_new

def calculate_mass_exchange(f_post, f_streamed, cell_type):
    def shift_fn_ct(shift):
        return jnp.roll(cell_type, shift=shift, axis=(0, 1))

    type_in = jax.vmap(shift_fn_ct)(VELOCITIES)
    type_out = jax.vmap(lambda shift: jnp.roll(cell_type, shift=-shift, axis=(0, 1)))(VELOCITIES)

    mass_in = jnp.where(type_in != TYPE_GAS, f_streamed, 0.0)
    mass_out = jnp.where(type_out != TYPE_GAS, f_post, 0.0)

    delta_m_dir = mass_in - mass_out
    delta_m = jnp.sum(delta_m_dir, axis=0)

    delta_m = jnp.where(cell_type == TYPE_INTERFACE, delta_m, 0.0)
    return delta_m

def compute_fraction(mass, rho, cell_type):
    eps = mass / rho
    eps = jnp.where(cell_type == TYPE_FLUID, 1.0, eps)
    eps = jnp.where(cell_type == TYPE_GAS, 0.0, eps)
    return jnp.clip(eps, 0.0, 1.0)

def update_topology(mass, rho, cell_type):
    new_type = cell_type

    filled_mask = (cell_type == TYPE_INTERFACE) & (mass >= rho)
    emptied_mask = (cell_type == TYPE_INTERFACE) & (mass <= 0.0)

    new_type = jnp.where(filled_mask, TYPE_FLUID, new_type)
    new_type = jnp.where(emptied_mask, TYPE_GAS, new_type)

    def shift_fn(shift):
        return jnp.roll(new_type, shift=shift, axis=(0, 1))

    neighbor_types = jax.vmap(shift_fn)(VELOCITIES)

    has_fluid_neighbor = jnp.any(neighbor_types == TYPE_FLUID, axis=0)
    has_gas_neighbor = jnp.any(neighbor_types == TYPE_GAS, axis=0)

    gas_to_interface = (new_type == TYPE_GAS) & has_fluid_neighbor
    fluid_to_interface = (new_type == TYPE_FLUID) & has_gas_neighbor

    final_type = new_type
    final_type = jnp.where(gas_to_interface, TYPE_INTERFACE, final_type)
    final_type = jnp.where(fluid_to_interface, TYPE_INTERFACE, final_type)

    final_mass = mass
    # simple clamp
    final_mass = jnp.where(filled_mask, rho, final_mass)
    final_mass = jnp.where(emptied_mask, 0.0, final_mass)

    final_mass = jnp.where(gas_to_interface, 0.1 * rho, final_mass)
    final_mass = jnp.where(fluid_to_interface, 0.9 * rho, final_mass)

    return final_mass, final_type

def init_new_interface_distributions(f, cell_type, old_cell_type):
    return f
