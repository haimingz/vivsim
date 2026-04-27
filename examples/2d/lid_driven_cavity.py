r"""
2D lid-driven cavity flow using LBM

This example can be used to test the stability of different collision models
and boundary conditions. Generally, the non-equilibrium extrapolation (NEE)
scheme is more stable than the non-equilibrium bounce-back (NEBB) scheme.
Adopting the KBC collision model also helps improve stability at the boundaries.

"""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

from vivsim import lbm, post


# ========================== VISUALIZATION PARAMETERS ====================

PLOT = True    # Enable chunked live plotting
CHUNK_STEPS = 500


# ========================== GEOMETRY =======================

NX = 1000      # Number of grid points in x-direction
NY = 1000      # Number of grid points in y-direction


# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.3          # Velocity of the moving lid
RE_GRID = 30       # Reynolds number based on grid size

# Derived physical parameters
NU = U0 / RE_GRID              # Kinematic viscosity
OMEGA = lbm.get_omega(NU)      # Relaxation parameter
TM = 80000         # Total simulation timesteps


# ======================= INITIALIZE VARIABLES ====================

rho = jnp.ones((NX, NY))           # Fluid density
u = jnp.zeros((2, NX, NY))         # Fluid velocity
f = lbm.get_equilibrium(rho, u)    # Initial distribution function


# ======================= SIMULATION ROUTINE =====================

# @partial(jax.jit, donate_argnums=0)
def update_step(f):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_bgk(f, feq, OMEGA)
    f = lbm.streaming(f)
    f = lbm.boundary_nee(f, loc="top", ux_wall=U0)
    f = lbm.boundary_nee(f, loc="left")
    f = lbm.boundary_nee(f, loc="right")
    f = lbm.boundary_nee(f, loc="bottom")
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
    return post.velocity_magnitude(u).T


# ======================= VISUALIZATION SETUP ====================

mpl.rcParams["figure.raise_window"] = False

if PLOT:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        get_plot_image(f),
        cmap="plasma_r", aspect="equal", origin="lower",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="$|u|$")
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
            plot_data = get_plot_image(f)
            im.set_data(plot_data)
            im.autoscale()
            plt.pause(0.001)


# ========================= POST-PROCESSING ==========================

_, u = lbm.get_macroscopic(f)
vel_mag = post.velocity_magnitude(u)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(
    vel_mag.T,
    cmap="plasma_r", aspect="equal", origin="lower",
)
ax.set_title("Lid-driven cavity: velocity magnitude")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(im, ax=ax, label="$|u|$")
fig.tight_layout()
plt.show()