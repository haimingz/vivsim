"""
Vortex-Induced Vibration (VIV) with a multi-block grid system

This example simulates the vortex-induced vibration of a circular cylinder
in a 2D flow using the Immersed Boundary Method (IBM) coupled with the
Lattice Boltzmann Method (LBM). The fluid domain is split into five blocks
with different refinement levels for improved efficiency.
"""

import math

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from vivsim import dyn, ib, lbm, post
from vivsim import multigrid as mg


# ========================== VISUALIZATION PARAMETERS ====================

PLOT = True  # Enable chunked live plotting (host sync on plot updates)
CHUNK_STEPS = 100

# ========================== GEOMETRY =======================

D = 48  # Cylinder diameter
NY = 20 * D  # Domain height (shared by all blocks)

BLOCK_WIDTHS = (8 * D, D // 2, 3 * D, D // 2, 18 * D)  # Width of each block
BLOCK_LEVELS = (-2, -1, 0, -1, -2)  # Refinement level of each block
BLOCK_EDGES = np.cumsum((0, *BLOCK_WIDTHS))

CYL_X = 10 * D - BLOCK_WIDTHS[0] - BLOCK_WIDTHS[1]  # Cylinder x-position in Block 2 coordinates
CYL_Y = 10 * D  # Cylinder y-position in Block 2 coordinates
CYL_AREA = math.pi * (D / 2) ** 2  # Cylinder cross-sectional area

N_MARKER = 4 * D  # Number of markers on cylinder surface
MARKER_THETA = jnp.linspace(0, 2 * jnp.pi, N_MARKER, dtype=jnp.float32, endpoint=False)
MARKER_X = CYL_X + 0.5 * D * jnp.cos(MARKER_THETA)  # Marker x-coordinates
MARKER_Y = CYL_Y + 0.5 * D * jnp.sin(MARKER_THETA)  # Marker y-coordinates
MARKER_COORDS = jnp.stack((MARKER_X, MARKER_Y), axis=1)
MARKER_DS = ib.get_ds_closed(MARKER_COORDS)  # Marker segment length


# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.05  # Inlet velocity
RE = 500  # Reynolds number
UR = 5  # Reduced velocity
MR = 10  # Mass ratio
DR = 0  # Damping ratio

NU = U0 * D / RE  # Kinematic viscosity
FN = U0 / (UR * D)  # Natural frequency of the structure
M = CYL_AREA * MR  # Mass of the cylinder per unit length
K = (2 * math.pi * FN) ** 2 * M * (1 + 1 / MR)  # Spring stiffness per unit length
C = 2 * math.sqrt(K * M) * DR  # Damping coefficient per unit length
TM = 60000  # Total simulation timesteps


# ========================== IB-LBM PARAMETERS ==========================

IB_PAD = 10  # Padding around cylinder for defining the local IBM region
IB_ITER = 3  # Multi-direct forcing iterations for IBM coupling
FSI_ITER = 1  # Number of fluid-structure iterations per timestep

IB_X0 = int(CYL_X - 0.5 * D - IB_PAD)  # X-coordinate of the initial IBM region
IB_Y0 = int(CYL_Y - 0.5 * D - IB_PAD)  # Y-coordinate of the initial IBM region
IB_SIZE = D + 2 * IB_PAD  # Size of the IBM region

OMEGA0 = mg.get_omega(NU, BLOCK_LEVELS[0])  # Block 0 relaxation parameter
OMEGA1 = mg.get_omega(NU, BLOCK_LEVELS[1])  # Block 1 relaxation parameter
OMEGA2 = mg.get_omega(NU, BLOCK_LEVELS[2])  # Block 2 relaxation parameter
OMEGA3 = mg.get_omega(NU, BLOCK_LEVELS[3])  # Block 3 relaxation parameter
OMEGA4 = mg.get_omega(NU, BLOCK_LEVELS[4])  # Block 4 relaxation parameter


# ======================= INITIALIZE VARIABLES ====================

f0, rho0, u0 = mg.init_grid(BLOCK_WIDTHS[0], NY, BLOCK_LEVELS[0], buffer_x=1)
f1, rho1, u1 = mg.init_grid(BLOCK_WIDTHS[1], NY, BLOCK_LEVELS[1], buffer_x=1)
f2, rho2, u2 = mg.init_grid(BLOCK_WIDTHS[2], NY, BLOCK_LEVELS[2])
f3, rho3, u3 = mg.init_grid(BLOCK_WIDTHS[3], NY, BLOCK_LEVELS[3], buffer_x=1)
f4, rho4, u4 = mg.init_grid(BLOCK_WIDTHS[4], NY, BLOCK_LEVELS[4], buffer_x=1)

u0 = u0.at[0].set(U0)
u1 = u1.at[0].set(U0)
u2 = u2.at[0].set(U0)
u3 = u3.at[0].set(U0)
u4 = u4.at[0].set(U0)

f0 = lbm.get_equilibrium(rho0, u0)
f1 = lbm.get_equilibrium(rho1, u1)
f2 = lbm.get_equilibrium(rho2, u2)
f3 = lbm.get_equilibrium(rho3, u3)
f4 = lbm.get_equilibrium(rho4, u4)

d = jnp.zeros(2, dtype=jnp.float32)  # Displacement of cylinder
v = jnp.zeros(2, dtype=jnp.float32)  # Velocity of cylinder
a = jnp.zeros(2, dtype=jnp.float32)  # Acceleration of cylinder
h = jnp.zeros(2, dtype=jnp.float32)  # Hydrodynamic force on the cylinder

# Optional: add a small cross-flow perturbation
v = v.at[1].set(1e-2 * U0)


# ======================= SIMULATION ROUTINES =====================

def solve_fsi(f, rho, u, d, v, a):

    # Dynamic IB region coordinates
    ib_x0 = (IB_X0 + d[0]).astype(jnp.int32)
    ib_y0 = (IB_Y0 + d[1]).astype(jnp.int32)

    # Extract IBM region data for efficient computation
    ib_rho = jax.lax.dynamic_slice(rho, (ib_x0, ib_y0), (IB_SIZE, IB_SIZE))
    ib_u = jax.lax.dynamic_slice(u, (0, ib_x0, ib_y0), (2, IB_SIZE, IB_SIZE))
    ib_f = jax.lax.dynamic_slice(f, (0, ib_x0, ib_y0), (9, IB_SIZE, IB_SIZE))

    a_old, v_old, d_old = a, v, d

    for _ in range(FSI_ITER):
        marker_x, marker_y = dyn.get_markers_coords_2dof(MARKER_X, MARKER_Y, d)

        stencil_weights, stencil_indices = ib.get_ib_stencil(
            marker_x=marker_x - ib_x0,
            marker_y=marker_y - ib_y0,
            ny=IB_SIZE,
            kernel=ib.kernel_peskin_4pt,
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
        a, v, d = dyn.newmark_2dof(a_old, v_old, d_old, h, M, K, C)

    ib_f = lbm.forcing_edm(ib_f, ib_g, ib_u, ib_rho)
    f = jax.lax.dynamic_update_slice(f, ib_f, (0, ib_x0, ib_y0))

    ib_u = ib_u + lbm.get_velocity_correction(ib_g, ib_rho)
    u = jax.lax.dynamic_update_slice(u, ib_u, (0, ib_x0, ib_y0))

    return f, u, d, v, a, h


def update_block2(f1, f2, f3, d, v, a):

    # Collision
    rho2, u2 = lbm.get_macroscopic(f2)
    feq2 = lbm.get_equilibrium(rho2, u2)
    f2 = lbm.collision_kbc(f2, feq2, OMEGA2)

    # Fluid-structure coupling in the finest block
    f2, u2, d, v, a, h = solve_fsi(f2, rho2, u2, d, v, a)

    # Streaming and inter-block boundary updates
    f2 = lbm.streaming(f2)
    f2 = mg.coarse_to_fine(f1, f2, dir="right")
    f2 = mg.coarse_to_fine(f3, f2, dir="left")

    return f2, rho2, u2, d, v, a, h


def update_level1(f0, f1, f2, f3, f4, d, v, a):

    # Collision on the intermediate blocks
    rho1, u1 = lbm.get_macroscopic(f1)
    feq1 = lbm.get_equilibrium(rho1, u1)
    f1 = lbm.collision_kbc(f1, feq1, OMEGA1)

    rho3, u3 = lbm.get_macroscopic(f3)
    feq3 = lbm.get_equilibrium(rho3, u3)
    f3 = lbm.collision_kbc(f3, feq3, OMEGA3)

    f1 = lbm.streaming(f1)
    f3 = lbm.streaming(f3)

    f2, rho2, u2, d, v, a, h = update_block2(f1, f2, f3, d, v, a)
    f2, rho2, u2, d, v, a, h = update_block2(f1, f2, f3, d, v, a)

    f1 = mg.fine_to_coarse(f2, f1, dir="left")
    f1 = mg.coarse_to_fine(f0, f1, dir="right")

    f3 = mg.fine_to_coarse(f2, f3, dir="right")
    f3 = mg.coarse_to_fine(f4, f3, dir="left")

    return f1, rho1, u1, f2, rho2, u2, f3, rho3, u3, d, v, a, h


@jax.jit
def update_level2(f0, f1, f2, f3, f4, d, v, a):

    # Collision on the coarsest blocks
    rho0, u0 = lbm.get_macroscopic(f0)
    feq0 = lbm.get_equilibrium(rho0, u0)
    f0 = lbm.collision_kbc(f0, feq0, OMEGA0)

    rho4, u4 = lbm.get_macroscopic(f4)
    feq4 = lbm.get_equilibrium(rho4, u4)
    f4 = lbm.collision_kbc(f4, feq4, OMEGA4)

    f0 = lbm.streaming(f0)
    f4 = lbm.streaming(f4)

    f1, rho1, u1, f2, rho2, u2, f3, rho3, u3, d, v, a, h = update_level1(f0, f1, f2, f3, f4, d, v, a)
    f1, rho1, u1, f2, rho2, u2, f3, rho3, u3, d, v, a, h = update_level1(f0, f1, f2, f3, f4, d, v, a)

    f0 = mg.fine_to_coarse(f1, f0, dir="left")
    f0 = lbm.boundary_nee(f0, loc="left", ux_wall=U0)

    f4 = mg.fine_to_coarse(f3, f4, dir="right")
    f4 = lbm.boundary_equilibrium(f4, loc="right", ux_wall=U0)

    return f0, rho0, u0, f1, rho1, u1, f2, rho2, u2, f3, rho3, u3, f4, rho4, u4, d, v, a, h


@jax.jit
def get_vorticity_image(u, scale):
    return post.calculate_curl(u).T / (D * U0 * scale)


# ======================= VISUALIZATION SETUP ====================

mpl.rcParams["figure.raise_window"] = False

d_hist = np.zeros((2, TM), dtype=np.float32)
h_hist = np.zeros((2, TM), dtype=np.float32)

if PLOT:
    plt.figure(figsize=(10, 4))

    kwargs = dict(
        cmap="seismic",
        aspect="equal",
        origin="lower",
        vmax=0.01,
        vmin=-0.01,
    )

    im0 = plt.imshow(
        get_vorticity_image(u0, scale=4),
        extent=[BLOCK_EDGES[0] / D, BLOCK_EDGES[1] / D, 0, NY / D],
        **kwargs,
    )
    im1 = plt.imshow(
        get_vorticity_image(u1, scale=2),
        extent=[BLOCK_EDGES[1] / D, BLOCK_EDGES[2] / D, 0, NY / D],
        **kwargs,
    )
    im2 = plt.imshow(
        get_vorticity_image(u2, scale=1),
        extent=[BLOCK_EDGES[2] / D, BLOCK_EDGES[3] / D, 0, NY / D],
        **kwargs,
    )
    im3 = plt.imshow(
        get_vorticity_image(u3, scale=2),
        extent=[BLOCK_EDGES[3] / D, BLOCK_EDGES[4] / D, 0, NY / D],
        **kwargs,
    )
    im4 = plt.imshow(
        get_vorticity_image(u4, scale=4),
        extent=[BLOCK_EDGES[4] / D, BLOCK_EDGES[5] / D, 0, NY / D],
        **kwargs,
    )

    plt.colorbar(label="Vorticity * D / U0")
    plt.xlabel("x / D")
    plt.ylabel("y / D")

    for block_edge in BLOCK_EDGES[1:-1]:
        plt.axvline(block_edge / D, color="g", linestyle="--", linewidth=0.5)

    plt.plot(
        (CYL_X + BLOCK_EDGES[2]) / D,
        CYL_Y / D,
        marker="+",
        markersize=10,
        color="k",
        linestyle="None",
        markeredgewidth=0.5,
    )

    plt.tight_layout()


# ========================== RUN SIMULATION ==========================

def run_chunk(carry, n_steps):
    def step(carry, _):
        f0, u0, f1, u1, f2, u2, f3, u3, f4, u4, d, v, a = carry
        f0, _, u0, f1, _, u1, f2, _, u2, f3, _, u3, f4, _, u4, d, v, a, h = update_level2(f0, f1, f2, f3, f4, d, v, a)
        return (f0, u0, f1, u1, f2, u2, f3, u3, f4, u4, d, v, a), (d, h)

    return jax.lax.scan(step, carry, None, length=n_steps)


run_chunk = jax.jit(run_chunk, static_argnums=1)

d_chunks = []
h_chunks = []

chunk_sizes = [CHUNK_STEPS] * (TM // CHUNK_STEPS)
if TM % CHUNK_STEPS:
    chunk_sizes.append(TM % CHUNK_STEPS)

step_count = 0

with tqdm(total=TM, unit="step") as pbar:
    for n_steps in chunk_sizes:
        (f0, u0, f1, u1, f2, u2, f3, u3, f4, u4, d, v, a), (d_chunk, h_chunk) = run_chunk(
            (f0, u0, f1, u1, f2, u2, f3, u3, f4, u4, d, v, a),
            n_steps,
        )
        jax.block_until_ready(d_chunk)

        d_chunks.append(d_chunk)
        h_chunks.append(h_chunk)

        step_count += n_steps
        pbar.update(n_steps)

        if PLOT:
            im0.set_data(get_vorticity_image(u0, scale=4))
            im1.set_data(get_vorticity_image(u1, scale=2))
            im2.set_data(get_vorticity_image(u2, scale=1))
            im3.set_data(get_vorticity_image(u3, scale=2))
            im4.set_data(get_vorticity_image(u4, scale=4))
            plt.pause(0.01)

d_hist = jnp.swapaxes(jnp.concatenate(d_chunks, axis=0), 0, 1)
h_hist = jnp.swapaxes(jnp.concatenate(h_chunks, axis=0), 0, 1)

if PLOT:
    plt.show()
