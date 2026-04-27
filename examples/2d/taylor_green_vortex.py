r"""
2D Taylor-Green vortex decay on a periodic domain.

This is a classic analytical benchmark for periodic LBM solvers. The initial
velocity field is divergence-free, and the exact solution decays exponentially
with a rate controlled by the viscosity and wave number.

"""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from vivsim import lbm, post


# ========================== SIMULATION PARAMETERS ====================

CHUNK_STEPS = 100
SAMPLE_EVERY = 100  # Sample kinetic energy every n steps


# ========================== GEOMETRY =======================

NX = 64       # Number of grid points in x-direction
NY = 64       # Number of grid points in y-direction

KX = 2 * jnp.pi / NX  # Wavenumber in x
KY = 2 * jnp.pi / NY  # Wavenumber in y
K2 = KX ** 2 + KY ** 2  # Squared wavenumber

X = jnp.arange(NX, dtype=jnp.float32)[:, None]
Y = jnp.arange(NY, dtype=jnp.float32)[None, :]


# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.02     # Velocity amplitude
NU = 0.02     # Kinematic viscosity

OMEGA = lbm.get_omega(NU)  # Relaxation parameter
TM = 4000     # Total simulation timesteps


# ======================= HELPER FUNCTIONS ====================

def exact_velocity(t):
    """Compute the analytical velocity field at time t."""
    decay = jnp.exp(-NU * K2 * t)
    ux = U0 * jnp.sin(KX * X) * jnp.cos(KY * Y) * decay
    uy = -U0 * jnp.cos(KX * X) * jnp.sin(KY * Y) * decay
    return jnp.stack((ux, uy))


def kinetic_energy(u):
    """Compute the mean kinetic energy of the velocity field."""
    return 0.5 * jnp.mean(jnp.sum(u ** 2, axis=0))


# ======================= INITIALIZE VARIABLES ====================

rho = jnp.ones((NX, NY))               # Fluid density
u = exact_velocity(0.0)                 # Initial velocity field (analytical)
f = lbm.get_equilibrium(rho, u)        # Initial distribution function


# ======================= SIMULATION ROUTINE =====================

def update_step(f):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_bgk(f, feq, OMEGA)
    f = lbm.streaming(f)
    return f


@partial(jax.jit, static_argnums=1, donate_argnums=0)
def update_chunk(carry, n_steps):
    def step(carry, _):
        (f,) = carry
        f = update_step(f)
        return (f,), None
    return jax.lax.scan(step, carry, None, length=n_steps)


# ========================== RUN SIMULATION ==========================

sample_steps = list(range(0, TM + 1, SAMPLE_EVERY))
energy_hist = [float(kinetic_energy(u))]

chunk_sizes = [CHUNK_STEPS] * (TM // CHUNK_STEPS)
if TM % CHUNK_STEPS:
    chunk_sizes.append(TM % CHUNK_STEPS)

current_step = 0
with tqdm(total=TM, unit="step") as pbar:
    for n_steps in chunk_sizes:
        (f,), _ = update_chunk((f,), n_steps)
        jax.block_until_ready(f)
        current_step += n_steps
        pbar.update(n_steps)

        # Sample energy at chunk boundaries that align with SAMPLE_EVERY
        if current_step % SAMPLE_EVERY == 0:
            _, u_now = lbm.get_macroscopic(f)
            energy_hist.append(float(kinetic_energy(u_now)))


# ========================= POST-PROCESSING ==========================

_, u = lbm.get_macroscopic(f)
u_true = exact_velocity(float(TM))
vorticity = post.vorticity(u)

rel_l2 = float(jnp.linalg.norm(u - u_true) / jnp.linalg.norm(u_true))

sample_steps_arr = np.arange(0, TM + 1, SAMPLE_EVERY)
energy_hist_arr = np.array(energy_hist[:len(sample_steps_arr)])
energy_true = energy_hist_arr[0] * np.exp(-2 * float(NU * K2) * sample_steps_arr)

slice_y = NY // 8
x = np.arange(NX)
grid_x, grid_y = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
quiver_stride = max(NX // 16, 1)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

axes[0].plot(sample_steps_arr, energy_true, label="Analytical", linewidth=2)
axes[0].plot(sample_steps_arr, energy_hist_arr, "o", ms=3, label="LBM")
axes[0].set_title("Kinetic energy decay")
axes[0].set_xlabel("Time step")
axes[0].set_ylabel("Energy")
axes[0].grid(alpha=0.3, linestyle=":")
axes[0].legend(frameon=False)

axes[1].plot(x, np.asarray(u_true[0, :, slice_y]), label="Analytical", linewidth=2)
axes[1].plot(x, np.asarray(u[0, :, slice_y]), "--", label="LBM")
axes[1].set_title(f"Final velocity slice error = {rel_l2:.2e}")
axes[1].set_xlabel("x")
axes[1].set_ylabel(f"$u_x(x, y={slice_y})$")
axes[1].grid(alpha=0.3, linestyle=":")
axes[1].legend(frameon=False)

vmax = float(jnp.max(jnp.abs(vorticity)))
im = axes[2].imshow(
    np.asarray(vorticity.T),
    origin="lower",
    cmap="RdBu_r",
    extent=[0, NX - 1, 0, NY - 1],
    vmin=-vmax,
    vmax=vmax,
)
axes[2].quiver(
    grid_x[::quiver_stride, ::quiver_stride],
    grid_y[::quiver_stride, ::quiver_stride],
    np.asarray(u[0, ::quiver_stride, ::quiver_stride].T),
    np.asarray(u[1, ::quiver_stride, ::quiver_stride].T),
    color="black", alpha=0.7,
    scale=0.12, scale_units="xy", angles="xy", width=0.004,
)
axes[2].set_title("Final vorticity field")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
fig.colorbar(im, ax=axes[2], label="$\\omega_z$")

fig.suptitle("2D Taylor-Green vortex decay")
fig.tight_layout()
plt.show()
