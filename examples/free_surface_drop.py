"""
This script demonstrates the Single-Phase Free-Surface LBM approach implemented in JAX.
Instead of solving a multi-phase flow using models like Shan-Chen, which requires solving
both fluid and gas phases, this approach explicitly tracks the fluid phase and a 1-cell thick
interface layer. The gas phase is simplified to a constant pressure boundary condition.

This is highly efficient for water waves, splashing, and other problems with high density ratios
where the gas dynamics can be neglected.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from vivsim import lbm

# Parameters
NX = 100
NY = 100
TM = 2000
NU = 0.05
OMEGA = lbm.get_omega(NU)
G = -0.00005

RHO_GAS = 1.0

# Initialize fluid domain
x, y = jnp.meshgrid(jnp.arange(NX), jnp.arange(NY), indexing='ij')

# Keep pool far from boundaries to observe physics
pool_mask = (y > 5) & (y < 30) & (x > 5) & (x < 95)
drop_mask = (x - 50)**2 + (y - 70)**2 < 15**2

fluid_mask = pool_mask | drop_mask

cell_type = jnp.full((NX, NY), lbm.TYPE_GAS)
cell_type = jnp.where(fluid_mask, lbm.TYPE_FLUID, cell_type)

def get_interfaces(ctype):
    def shift_fn(shift):
        return jnp.roll(ctype, shift=shift, axis=(0, 1))
    neighbor_types = jax.vmap(shift_fn)(lbm.VELOCITIES)
    has_fluid = jnp.any(neighbor_types == lbm.TYPE_FLUID, axis=0)
    return (ctype == lbm.TYPE_GAS) & has_fluid

cell_type = jnp.where(get_interfaces(cell_type), lbm.TYPE_INTERFACE, cell_type)

rho = jnp.where(cell_type != lbm.TYPE_GAS, 1.0, RHO_GAS)
u = jnp.zeros((2, NX, NY))

mass = jnp.where(cell_type == lbm.TYPE_FLUID, 1.0, 0.0)
mass = jnp.where(cell_type == lbm.TYPE_INTERFACE, 0.5, mass)

f = lbm.get_equilibrium(rho, u)

@jax.jit
def update(f, mass, cell_type):
    # 1. Macroscopic variables
    rho_fluid, u_fluid = lbm.get_macroscopic(f)
    g_force = jnp.array([0.0, G])[:, None, None] * rho_fluid
    u_fluid = u_fluid + lbm.get_velocity_correction(g_force, rho_fluid)

    u = jnp.where(cell_type[None, ...] == lbm.TYPE_GAS, 0.0, u_fluid)
    rho_curr = jnp.where(cell_type == lbm.TYPE_GAS, RHO_GAS, rho_fluid)

    # 2. Collision & Forcing
    feq = lbm.get_equilibrium(rho_curr, u)
    f_post = lbm.collision_bgk(f, feq, OMEGA)

    f_post_f = lbm.forcing_edm(f_post, g_force, u, rho_curr)
    f_post = jnp.where(cell_type[None, ...] != lbm.TYPE_GAS, f_post_f, f_post)

    # 3. Stream
    f_streamed = lbm.streaming(f_post)

    # Apply solid boundaries (bounce back)
    f_streamed = lbm.boundary_bounce_back(f_post, f_streamed, loc='bottom')
    f_streamed = lbm.boundary_bounce_back(f_post, f_streamed, loc='left')
    f_streamed = lbm.boundary_bounce_back(f_post, f_streamed, loc='right')

    # 4. Reconstruct unknown distributions at interface from gas pressure
    f_new = lbm.reconstruct_interface_distributions(f_post, f_streamed, cell_type, RHO_GAS, u)

    # 5. Mass exchange
    delta_m = lbm.calculate_mass_exchange(f_post, f_streamed, cell_type)

    # Zero out mass exchange at boundaries to avoid mass leaking through periodic rolls
    delta_m = delta_m.at[0, :].set(0.0)
    delta_m = delta_m.at[-1, :].set(0.0)
    delta_m = delta_m.at[:, 0].set(0.0)
    delta_m = delta_m.at[:, -1].set(0.0)

    mass_new = mass + delta_m

    # 6. Topology update (converting full interfaces to fluid, empty to gas)
    mass_new, cell_type_new = lbm.update_topology(mass_new, rho_curr, cell_type)

    # Keep Gas cells at constant equilibrium
    f_new = jnp.where(cell_type_new[None, ...] == lbm.TYPE_GAS, lbm.get_equilibrium(jnp.full((NX, NY), RHO_GAS), jnp.zeros((2, NX, NY))), f_new)

    return f_new, mass_new, cell_type_new

history_ctype = []

for t in tqdm(range(TM)):
    f, mass, cell_type = update(f, mass, cell_type)
    if t % 250 == 0:
        history_ctype.append(cell_type.copy())

fig, axes = plt.subplots(1, len(history_ctype), figsize=(15, 3))
for i, c in enumerate(history_ctype):
    axes[i].imshow(c.T, origin='lower', cmap='Blues', vmin=-1, vmax=1)
    axes[i].axis('off')
    axes[i].set_title(f"t={i*250}")

plt.tight_layout()
plt.savefig('free_surface_drop.png')
print("Saved to free_surface_drop.png")
