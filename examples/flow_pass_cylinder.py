"""
Flow past a stationary circular cylinder using IB-LBM

This example simulates incompressible flow past a fixed cylinder and can be
used to validate the IB-LBM implementation by comparing the hydrodynamic force
coefficients against reference data from the literature.
"""

import math

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from vivsim import dyn, ib, lbm


# ========================== GEOMETRY =======================

D = 60  # Cylinder diameter
NX = 30 * D  # Domain width
NY = 20 * D  # Domain height
CYL_X = 10 * D  # Cylinder center x-position
CYL_Y = 10 * D  # Cylinder center y-position
CYL_AREA = math.pi * (D / 2) ** 2  # Cylinder cross-sectional area

N_MARKER = 4 * D  # Number of markers on cylinder surface
MARKER_THETA = jnp.linspace(0, 2 * jnp.pi, N_MARKER, endpoint=False)
MARKER_X = CYL_X + 0.5 * D * jnp.cos(MARKER_THETA)  # Marker x-coordinates
MARKER_Y = CYL_Y + 0.5 * D * jnp.sin(MARKER_THETA)  # Marker y-coordinates
MARKER_COORDS = jnp.stack((MARKER_X, MARKER_Y), axis=1)
MARKER_DS = ib.get_ds_closed(MARKER_COORDS)  # Marker segment length


# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.05  # Inlet velocity
RE = 200  # Reynolds number

NU = U0 * D / RE  # Kinematic viscosity
TM = int(30 * 5 / U0 * D)  # Total simulation timesteps


# ========================== IB-LBM PARAMETERS ==========================

OMEGA = lbm.get_omega(NU)  # Relaxation parameter

IB_ITER = 3  # Multi-direct forcing iterations for IBM coupling
IB_PAD = 2  # Padding around cylinder for defining the local IBM region

IB_X0 = int(CYL_X - 0.5 * D - IB_PAD)  # X-coordinate of the IBM region
IB_Y0 = int(CYL_Y - 0.5 * D - IB_PAD)  # Y-coordinate of the IBM region
IB_SIZE = D + 2 * IB_PAD  # Size of the IBM region

# Pre-compute marker stencils in the local IBM region.
MARKER_X_IB = MARKER_X - IB_X0
MARKER_Y_IB = MARKER_Y - IB_Y0
STENCIL_WEIGHTS, STENCIL_INDICES = ib.get_ib_stencil(
    MARKER_X_IB, MARKER_Y_IB,
    IB_SIZE,
    kernel=ib.kernel_peskin_4pt,
)


# ========================== VISUALIZATION PARAMETERS ====================

CHUNK_STEPS = 500  # Steps per scan chunk


# ========================== REFERENCE DATA ==========================

# Averaged from the following sources:
# [1] https://doi.org/10.1016/j.ijmecsci.2024.108961
# [2] https://doi.org/10.1016/j.camwa.2014.05.013
# [3] https://doi.org/10.1017/S0022112086003014
CD_REF = 1.389  # Mean drag coefficient
CL_REF = 0.718  # Maximum lift coefficient amplitude


# ======================= INITIALIZE VARIABLES ====================

rho = jnp.ones((NX, NY))  # Fluid density
u = jnp.zeros((2, NX, NY)).at[0].set(U0)  # Uniform inlet flow
f = lbm.get_equilibrium(rho, u)  # Initial LBM distribution function

marker_v = jnp.zeros((N_MARKER, 2))  # Target marker velocity for a fixed cylinder


# ======================= SIMULATION ROUTINE =====================

@jax.jit
def update(f):

    # Collision
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)

    # Extract IBM region data for efficient computation
    ib_rho = jax.lax.dynamic_slice(rho, (IB_X0, IB_Y0), (IB_SIZE, IB_SIZE))
    ib_u = jax.lax.dynamic_slice(u, (0, IB_X0, IB_Y0), (2, IB_SIZE, IB_SIZE))
    ib_f = jax.lax.dynamic_slice(f, (0, IB_X0, IB_Y0), (9, IB_SIZE, IB_SIZE))

    # Run multi-direct forcing to enforce the no-slip condition
    ib_g, marker_h = ib.multi_direct_forcing(
        grid_u=ib_u,
        stencil_weights=STENCIL_WEIGHTS,
        stencil_indices=STENCIL_INDICES,
        marker_u_target=marker_v,
        marker_ds=MARKER_DS,
        n_iter=IB_ITER,
    )

    h = dyn.get_force_to_obj(marker_h)

    # Apply IBM forcing to the fluid in the local IBM region
    ib_f = lbm.forcing_edm(ib_f, ib_g, ib_u, ib_rho)
    f = jax.lax.dynamic_update_slice(f, ib_f, (0, IB_X0, IB_Y0))

    # Streaming and boundary conditions
    f = lbm.streaming(f)
    f = lbm.boundary_nebb(f, loc="left", ux_wall=U0)
    f = lbm.boundary_equilibrium(f, loc="right", ux_wall=U0)

    return f, h


# ======================= CHUNKED SIMULATION WITH SCAN =====================

def run_chunk(carry, n_steps):
    def step(f, _):
        f, h = update(f)
        return f, h
    f, h_chunk = jax.lax.scan(step, carry, None, length=n_steps)
    return f, h_chunk

run_chunk = jax.jit(run_chunk, static_argnums=1)

chunk_sizes = [CHUNK_STEPS] * (TM // CHUNK_STEPS)
if TM % CHUNK_STEPS:
    chunk_sizes.append(TM % CHUNK_STEPS)

h_chunks = []

with tqdm(total=TM, unit="step") as pbar:
    for n_steps in chunk_sizes:
        f, h_chunk = run_chunk(f, n_steps)
        jax.block_until_ready(h_chunk)
        h_chunks.append(h_chunk)
        pbar.update(n_steps)

h_hist = np.asarray(jnp.concatenate(h_chunks, axis=0).T)


# ======================= POST-PROCESSING ==========================

cd = h_hist[0] * 2 / (D * U0 ** 2)
cl = h_hist[1] * 2 / (D * U0 ** 2)

cd_mean = np.mean(cd[TM // 2 :])
cl_max = np.max(np.abs(cl[TM // 2 :]))
cd_err = abs(cd_mean - CD_REF) / CD_REF * 100
cl_err = abs(cl_max - CL_REF) / CL_REF * 100

mpl.rcParams["figure.raise_window"] = False

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cd, label="Cd (drag)", linewidth=1.5)
ax.plot(cl, label="Cl (lift)", linewidth=1.5)

ax.set_xlabel("Time step")
ax.set_ylabel("Force coefficient")
ax.legend(frameon=False, loc="upper right")
ax.set_ylim(-2, 4)
ax.grid(True, alpha=0.3, linestyle=":")

stats_text = (
    f"Cd = {cd_mean:.3f} (ref: {CD_REF}, err: {cd_err:.1f}%)\n"
    f"Cl = {cl_max:.3f} (ref: {CL_REF}, err: {cl_err:.1f}%)"
)
ax.text(
    0.02,
    0.95,
    stats_text,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9,
    color="blue",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

fig.tight_layout()
plt.show()
