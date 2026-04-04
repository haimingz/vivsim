"""
Vortex-Induced Vibration (VIV) of a circular cylinder using IB-LBM

This example simulates the vortex-induced vibration of a circular cylinder
in a 2D flow using the Immersed Boundary Method (IBM) coupled with the
Lattice Boltzmann Method (LBM). The cylinder is free to oscillate in both
cross-flow and in-line directions with spring-mass-damper constraints.
"""

import math

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from vivsim import dyn, ib, lbm, post
from vivsim.state import FSIState


# ========================== VISUALIZATION PARAMETERS ====================

PLOT = True   # Enable chunked live plotting (host sync on plot updates)
CHUNK_STEPS = 300


# ========================== GEOMETRY =======================

D = 32          # Cylinder diameter
NX = 20 * D     # Domain width
NY = 10 * D     # Domain height
CYL_X = 5 * D   # Cylinder center x-position
CYL_Y = 5 * D   # Cylinder center y-position
CYL_AREA = math.pi * (D / 2) ** 2  # Cylinder cross-sectional area

N_MARKER = 4 * D  # Number of markers on cylinder surface
MARKER_THETA = jnp.linspace(0, 2 * jnp.pi, N_MARKER, endpoint=False)
MARKER_X = CYL_X + 0.5 * D * jnp.cos(MARKER_THETA)  # Marker x-coordinates
MARKER_Y = CYL_Y + 0.5 * D * jnp.sin(MARKER_THETA)  # Marker y-coordinates
MARKER_DS = 2 * math.pi * (D / 2) / N_MARKER  # Marker segment length

# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.05     # Inlet velocity
RE = 150      # Reynolds number
UR = 5        # Reduced velocity (U0 / (FN * D))
MR = 10       # Mass ratio (cylinder mass / displaced fluid mass)
DR = 0        # Damping ratio

# Derived physical parameters
NU = U0 * D / RE           # Kinematic viscosity
FN = U0 / (UR * D)         # Natural frequency of the structure
M = CYL_AREA * MR          # Mass of the cylinder per unit length
K = (2 * math.pi * FN) ** 2 * M * (1 + 1 / MR) # Spring stiffness per unit length
C = 2 * math.sqrt(K * M) * DR # Damping coefficient per unit length

TM = int(30 / FN) # Total simulation timesteps, simulating n natural periods


# ========================== IB-LBM PARAMETERS ==========================

OMEGA = lbm.get_omega(NU)  # Relaxation parameter for LBM

IB_ITER = 1  # Multi-direct forcing iterations for IBM coupling
IB_PAD = 10  # Padding around cylinder for defining the local IBM region
FSI_ITER = 1  # Number of iterations for fluid-structure coupling within each timestep

IB_X0 = int(CYL_X - 0.5 * D - IB_PAD)  # X-coordinate of the initial IBM region
IB_Y0 = int(CYL_Y - 0.5 * D - IB_PAD)  # Y-coordinate of the initial IBM region
IB_SIZE = D + 2 * IB_PAD  # Size of the IBM region


# ======================= INITIALIZE VARIABLES ====================

rho = jnp.ones((NX, NY))
u = jnp.zeros((2, NX, NY)).at[0].set(U0)
f = lbm.get_equilibrium(rho, u)

state = FSIState(
    f=f,
    d=jnp.zeros(2),
    v=jnp.zeros(2).at[1].set(1e-2 * U0),  # small cross-flow perturbation
    a=jnp.zeros(2),
)


# ======================= SIMULATION ROUTINE =====================

@jax.jit
def update(state: FSIState):

    f, d, v, a = state.f, state.d, state.v, state.a

    # Collision
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)

    # Dynamic IB region coordinates
    ib_x0 = (IB_X0 + d[0]).astype(jnp.int32)
    ib_y0 = (IB_Y0 + d[1]).astype(jnp.int32)

    # Extract IBM region data for efficient computation
    ib_rho = jax.lax.dynamic_slice(rho, (ib_x0, ib_y0), (IB_SIZE, IB_SIZE))
    ib_u = jax.lax.dynamic_slice(u, (0, ib_x0, ib_y0), (2, IB_SIZE, IB_SIZE))
    ib_f = jax.lax.dynamic_slice(f, (0, ib_x0, ib_y0), (9, IB_SIZE, IB_SIZE))

    # Run multi-direct forcing to get IBM forces based on current marker positions
    a_old, v_old, d_old = a, v, d

    for _ in range(FSI_ITER):
        marker_x, marker_y = dyn.get_markers_coords_2dof(MARKER_X, MARKER_Y, d)

        stencil_weights, stencil_indices = ib.get_ib_stencil(
            marker_x=marker_x - ib_x0,
            marker_y=marker_y - ib_y0,
            ny=IB_SIZE,
            kernel=ib.kernel_peskin_4pt,
            stencil_radius=2,
        )
        marker_v = jnp.repeat(v[None, :], N_MARKER, axis=0)

        ib_g, marker_h = ib.multi_direct_forcing(
            grid_u=ib_u,
            stencil_weights=stencil_weights,
            stencil_indices=stencil_indices,
            marker_u_target=marker_v,
            marker_ds=MARKER_DS,
            n_iter=IB_ITER,
        )

        h = dyn.get_force_to_obj(marker_h)
        h += a * CYL_AREA
        a, v, d = dyn.newmark(a_old, v_old, d_old, h, M, K, C)

    # apply IBM forcing to the fluid in the local IBM region
    ib_f = lbm.forcing_edm(ib_f, ib_g, ib_u, ib_rho)
    f = jax.lax.dynamic_update_slice(f, ib_f, (0, ib_x0, ib_y0))

    # streaming and boundary conditions
    f = lbm.streaming(f)
    f = lbm.boundary_nebb(f, loc="left", ux_wall=U0)
    f = lbm.boundary_equilibrium(f, loc="right", ux_wall=U0)

    new_state = FSIState(f=f, d=d, v=v, a=a)
    return new_state, h

@jax.jit
def get_vorticity_image(f):
    _, u = lbm.get_macroscopic(f)
    return post.calculate_vorticity_dimensionless(u, D, U0).T


# ======================= VISUALIZATION SETUP ====================

mpl.rcParams["figure.raise_window"] = False

if PLOT:
    plt.figure(figsize=(8, 3))

    im = plt.imshow(
        get_vorticity_image(state.f),
        extent=[0, NX / D, 0, NY / D],
        cmap="bwr", aspect="equal", origin="lower",
        vmax=10, vmin=-10,
    )

    plt.colorbar(label="Vorticity * D / U0")
    plt.xlabel("x / D")
    plt.ylabel("y / D")

    # mark initial cylinder center
    plt.plot(CYL_X / D, CYL_Y / D,
        marker="+", markersize=10, color="k", linestyle="None",markeredgewidth=0.5,)

    plt.tight_layout()


# ========================== RUN SIMULATION ==========================

def run_chunk(state, n_steps):
    def step(state, _):
        state, h = update(state)
        return state, (state.d, h)
    return jax.lax.scan(step, state, None, length=n_steps)

run_chunk = jax.jit(run_chunk, static_argnums=1)


d_chunks = []
h_chunks = []

chunk_sizes = [CHUNK_STEPS] * (TM // CHUNK_STEPS)
if TM % CHUNK_STEPS:
    chunk_sizes.append(TM % CHUNK_STEPS)

with tqdm(total=TM, unit="step") as pbar:
    for n_steps in chunk_sizes:
        state, (d_chunk, h_chunk) = run_chunk(state, n_steps)
        jax.block_until_ready(d_chunk)

        d_chunks.append(d_chunk)
        h_chunks.append(h_chunk)

        pbar.update(n_steps)

        if PLOT:
            im.set_data(get_vorticity_image(state.f))
            plt.pause(0.001)

d_hist = jnp.swapaxes(jnp.concatenate(d_chunks, axis=0), 0, 1)
h_hist = jnp.swapaxes(jnp.concatenate(h_chunks, axis=0), 0, 1)

# ======================= POST-PROCESSING ==========================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

t_normalized = np.arange(TM) * FN

ax1.plot(t_normalized, d_hist[0] / D, label="x / D", linewidth=1)
ax1.plot(t_normalized, d_hist[1] / D, label="y / D", linewidth=1)
ax1.set_xlabel("Time step")
ax1.set_ylabel("Displacement / D")
ax1.legend(loc="upper left", ncol=2)

cd = h_hist[0] * 2 / (D * U0 ** 2)
cl = h_hist[1] * 2 / (D * U0 ** 2)

ax2.plot(t_normalized, cd, label="Cd (drag)", linewidth=1)
ax2.plot(t_normalized, cl, label="Cl (lift)", linewidth=1)
ax2.set_xlabel("Time step")
ax2.set_ylabel("Force coefficient")
ax2.legend(loc="upper left", ncol=2)
ax2.set_ylim(-2, 4)

fig.tight_layout()
plt.show()
