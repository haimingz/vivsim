r"""
Plane Couette flow in a 2D channel driven by a moving lid.

This classic benchmark complements the existing cavity and Poiseuille examples:
the streamwise direction is periodic, the lower wall is stationary, and the
upper wall moves with a constant velocity. The steady analytical solution is a
linear velocity profile.

"""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from vivsim import lbm


# ========================== SIMULATION PARAMETERS ====================

CHUNK_STEPS = 1000


# ========================== GEOMETRY =======================

NX = 16        # Number of grid points in the x-direction (periodic)
NY = 16        # Number of grid points in the y-direction (wall-bounded)


# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.08     # Velocity of the moving lid
NU = 0.10     # Kinematic viscosity

OMEGA = lbm.get_omega(NU)  # Relaxation parameter
TM = 12000    # Total simulation timesteps


# ======================= INITIALIZE VARIABLES ====================

rho = jnp.ones((NX, NY))           # Fluid density
u = jnp.zeros((2, NX, NY))         # Fluid velocity
f = lbm.get_equilibrium(rho, u)    # Initial distribution function


# ======================= SIMULATION ROUTINE =====================

def update_step(f):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_bgk(f, feq, OMEGA)
    f = lbm.streaming(f)
    f = lbm.boundary_nee(f, loc="bottom")
    f = lbm.boundary_nee(f, loc="top", ux_wall=U0)
    return f


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
ux = u[0]

y = jnp.arange(NY, dtype=jnp.float32)
ux_true = U0 * y / (NY - 1)
ux_mean = jnp.mean(ux, axis=0)
ux_true_2d = jnp.broadcast_to(ux_true, (NX, NY))
rel_l2 = jnp.linalg.norm(ux - ux_true_2d) / jnp.linalg.norm(ux_true_2d)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(np.asarray(ux_true), np.asarray(y), label="Analytical", linewidth=1)
ax.plot(np.asarray(ux_mean), np.asarray(y), "o", ms=3, label="LBM")
ax.set_title(f"Profile error = {float(rel_l2):.2e}")
ax.set_xlabel("$u_x$")
ax.set_ylabel("y")
ax.grid(alpha=0.3, linestyle=":")
ax.legend(frameon=False)
fig.suptitle("Couette flow: velocity profile")
fig.tight_layout()
plt.show()
