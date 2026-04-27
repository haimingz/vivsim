r"""
Poiseuille flow in a 2D channel driven by a constant body force using LBM

This example demonstrates different LBM collision models and forcing schemes.
The results are compared with the analytical solution to validate the
implementation. Four collision/forcing combinations are provided:

  - BGK + EDM (Exact Difference Method)
  - BGK + Guo forcing
  - MRT + Guo forcing
  - KBC + EDM

"""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from vivsim import lbm


# ========================== SIMULATION PARAMETERS ====================

CHUNK_STEPS = 10000


# ========================== GEOMETRY =======================

NX = 10       # Number of grid points in x-direction (periodic)
NY = 10       # Number of grid points in y-direction (wall-bounded)


# ========================== PHYSICAL PARAMETERS =====================

NU = 0.2      # Kinematic viscosity
GX = 0.001    # Body force in the x-direction

OMEGA = lbm.get_omega(NU)  # Relaxation parameter
TM = 100000   # Total simulation timesteps


# ========================== LBM PARAMETERS ==========================

# MRT precomputation (needed for MRT recipes)
MRT_COLLISION = lbm.get_mrt_collision_operator(OMEGA)
MRT_FORCING = lbm.get_mrt_forcing_operator(OMEGA)

# Since the external force is present at the wall, the velocity correction
# must be applied in the boundary condition step. Pre-compute the corrected
# (no-slip) wall velocity once.
# UX_WALL, _ = lbm.get_corrected_wall_velocity(0, 0, gx_wall=GX)


# ======================= INITIALIZE VARIABLES ====================

rho = jnp.ones((NX, NY))           # Fluid density
u = jnp.zeros((2, NX, NY))         # Fluid velocity

g = jnp.zeros((2, NX, NY))         # Body force field
g = g.at[0].set(GX)

f = lbm.get_equilibrium(rho, u - g / 2)  # Initial distribution function


# ======================= SIMULATION ROUTINE =====================

# The following four recipes demonstrate different collision/forcing
# combinations. Uncomment the desired recipe in the RUN SIMULATION section.

# Note: the velocity correction (get_velocity_correction) is necessary when
# external forces are present, due to the redefinition of the distribution
# function for second-order accuracy. It is independent of the collision
# operator and forcing scheme used.


def update_step_bgk_edm(f):
    """BGK collision with Exact Difference Method (EDM) forcing.

    EDM requires collision and forcing to use the uncorrected velocity;
    the correction is applied AFTER the collision & forcing step.
    """
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_bgk(f, feq, OMEGA)
    f = lbm.forcing_edm(f, g, u, rho)
    f = lbm.streaming(f)
    f = lbm.boundary_force_corrected_nebb(f, loc="top", gx_wall=GX)
    f = lbm.boundary_force_corrected_nebb(f, loc="bottom", gx_wall=GX)
    return f


def update_step_bgk_guo(f):
    """BGK collision with Guo forcing.

    Guo forcing requires collision and forcing to use the corrected velocity;
    the correction is applied BEFORE the collision & forcing step.
    """
    rho, u = lbm.get_macroscopic(f)
    u = u + lbm.get_velocity_correction(g, rho)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_bgk(f, feq, OMEGA)
    f = lbm.forcing_guo_bgk(f, g, u, OMEGA)
    f = lbm.streaming(f)
    f = lbm.boundary_force_corrected_nebb(f, loc="top", gx_wall=GX)
    f = lbm.boundary_force_corrected_nebb(f, loc="bottom", gx_wall=GX)
    return f


def update_step_mrt_guo(f):
    """MRT collision with Guo forcing.

    Guo forcing requires collision and forcing to use the corrected velocity;
    the correction is applied BEFORE the collision & forcing step.
    (This recipe is not as accurate as the other recipes.)
    """
    rho, u = lbm.get_macroscopic(f)
    u = u + lbm.get_velocity_correction(g, rho)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_mrt(f, feq, MRT_COLLISION)
    f = lbm.forcing_guo_mrt(f, g, u, MRT_FORCING)
    f = lbm.streaming(f)
    f = lbm.boundary_force_corrected_nebb(f, loc="top", gx_wall=GX)
    f = lbm.boundary_force_corrected_nebb(f, loc="bottom", gx_wall=GX)
    return f


def update_step_kbc_edm(f):
    """KBC collision with Exact Difference Method (EDM) forcing.

    EDM requires collision and forcing to use the uncorrected velocity;
    the correction is applied AFTER the collision & forcing step.
    """
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)
    f = lbm.forcing_edm(f, g, u, rho)
    f = lbm.streaming(f)
    f = lbm.boundary_force_corrected_nebb(f, loc="top", gx_wall=GX)
    f = lbm.boundary_force_corrected_nebb(f, loc="bottom", gx_wall=GX)
    return f


# Select the active recipe here:
update_step = update_step_kbc_edm


@partial(jax.jit, static_argnums=1, donate_argnums=0)
def update_chunk(carry, n_steps):
    def step(carry, _):
        (f,) = carry
        f = update_step(f)
        return (f,), None
    return jax.lax.scan(step, carry, None, length=n_steps)


# ========================== RUN SIMULATION ==========================

chunk_sizes = [CHUNK_STEPS] * (TM // CHUNK_STEPS)
if TM % CHUNK_STEPS:
    chunk_sizes.append(TM % CHUNK_STEPS)

with tqdm(total=TM, unit="step") as pbar:
    for n_steps in chunk_sizes:
        (f,), _ = update_chunk((f,), n_steps)
        jax.block_until_ready(f)
        pbar.update(n_steps)


# ========================= POST-PROCESSING ==========================

rho, u = lbm.get_macroscopic(f)
u = u + lbm.get_velocity_correction(g, rho)

H = NY - 1
y_ana = jnp.arange(0, H, 0.01)
ux_true = GX / 2 / NU * (y_ana * (H - y_ana))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.asarray(y_ana), np.asarray(ux_true), "-", label="Analytical")
ax.plot(np.arange(NY), np.asarray(u[0, NX // 2]), "+", label="LBM")
ax.set_xlabel("$y$")
ax.set_ylabel("$u_x$")
ax.set_title("Poiseuille flow in a channel")
ax.legend(frameon=False)
ax.grid(alpha=0.3, linestyle=":")
fig.tight_layout()
plt.show()
