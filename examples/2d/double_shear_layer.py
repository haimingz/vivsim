r"""
2D double shear layer roll-up on a periodic domain.

This classic benchmark starts from two thin, oppositely signed shear layers
with a small transverse perturbation. As the simulation evolves, the layers
become unstable and roll up into vortical structures.

"""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from vivsim import lbm, post


# ========================== VISUALIZATION PARAMETERS ====================

PLOT = True    # Enable chunked live plotting
CHUNK_STEPS = 100


# ========================== GEOMETRY =======================

NX = 128       # Number of grid points in x-direction
NY = 128       # Number of grid points in y-direction


# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.04     # Shear velocity amplitude
NU = 0.002    # Kinematic viscosity

SHEAR_STEEPNESS = 80.0     # Steepness of the shear layer profile
PERTURBATION = 0.05 * U0   # Transverse perturbation amplitude

OMEGA = lbm.get_omega(NU)  # Relaxation parameter
TM = 10000    # Total simulation timesteps


# ======================= INITIALIZE VARIABLES ====================

X = (jnp.arange(NX, dtype=jnp.float32) + 0.5) / NX
Y = (jnp.arange(NY, dtype=jnp.float32) + 0.5) / NY
X_GRID = X[:, None]
Y_GRID = Y[None, :]

upper_layer = jnp.tanh(SHEAR_STEEPNESS * (Y_GRID - 0.25))
lower_layer = jnp.tanh(SHEAR_STEEPNESS * (0.75 - Y_GRID))
ux0 = U0 * jnp.where(Y_GRID <= 0.5, upper_layer, lower_layer)
ux0 = jnp.broadcast_to(ux0, (NX, NY))
uy0 = PERTURBATION * jnp.sin(2 * jnp.pi * X_GRID)
uy0 = jnp.broadcast_to(uy0, (NX, NY))

rho = jnp.ones((NX, NY))                      # Fluid density
u = jnp.stack((ux0, uy0))                     # Initial velocity field
f = lbm.get_equilibrium(rho, u)               # Initial distribution function


# ======================= SIMULATION ROUTINE =====================

def update_step(f):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)
    f = lbm.streaming(f)
    return f


@partial(jax.jit, static_argnums=1, donate_argnums=0)
def update_chunk(carry, n_steps):
    def step(carry, _):
        (f,) = carry
        f = update_step(f)
        return (f,), None
    return jax.lax.scan(step, carry, None, length=n_steps)


@jax.jit
def get_plot_image(f):
    _, u = lbm.get_macroscopic(f)
    return post.vorticity(u).T


# ======================= VISUALIZATION SETUP ====================

mpl.rcParams["figure.raise_window"] = False

if PLOT:
    vort0 = post.vorticity(u)
    vort_lim = float(jnp.max(jnp.abs(vort0)))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(
        np.asarray(vort0.T),
        origin="lower",
        cmap="RdBu_r",
        extent=[0, NX - 1, 0, NY - 1],
        vmin=-vort_lim,
        vmax=vort_lim,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="$\\omega_z$")
    fig.tight_layout()


# ========================== RUN SIMULATION ==========================

chunk_sizes = [CHUNK_STEPS] * (TM // CHUNK_STEPS)
if TM % CHUNK_STEPS:
    chunk_sizes.append(TM % CHUNK_STEPS)

with tqdm(total=TM, unit="step") as pbar:
    for n_steps in chunk_sizes:
        (f,), _ = update_chunk((f,), n_steps)
        jax.block_until_ready(f)
        pbar.update(n_steps)

        if PLOT:
            vort = get_plot_image(f)
            im.set_data(np.asarray(vort))
            vort_lim = float(jnp.max(jnp.abs(vort)))
            im.set_clim(vmin=-vort_lim, vmax=vort_lim)
            plt.pause(0.001)
