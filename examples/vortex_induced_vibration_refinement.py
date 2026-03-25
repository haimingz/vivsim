"""
Vortex-Induced Vibration (VIV) with nested grid refinement

This example simulates the vortex-induced vibration of a circular cylinder
in a 2D flow using the Immersed Boundary Method (IBM) coupled with the
Lattice Boltzmann Method (LBM). Three nested grids with different refinement
levels are used to concentrate resolution around the moving cylinder.
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

D = 40  # Cylinder diameter
CYL_X = 10 * D  # Cylinder center x-position in the global domain
CYL_Y = 10 * D  # Cylinder center y-position in the global domain
CYL_AREA = math.pi * (D / 2) ** 2  # Cylinder cross-sectional area

N_MARKER = 4 * D  # Number of markers on cylinder surface

GRID1_LEVEL = -2  # Coarsest grid refinement level
GRID1_NX = 30 * D  # Coarsest grid width
GRID1_NY = 20 * D  # Coarsest grid height

GRID2_LEVEL = -1  # Intermediate grid refinement level
GRID2_NX = 12 * D  # Intermediate grid width
GRID2_NY = 8 * D  # Intermediate grid height
GRID2_X0 = 6 * D  # Left boundary in global coordinates
GRID2_X1 = GRID2_X0 + GRID2_NX  # Right boundary in global coordinates
GRID2_Y0 = (GRID1_NY - GRID2_NY) // 2  # Bottom boundary in global coordinates
GRID2_Y1 = GRID2_Y0 + GRID2_NY  # Top boundary in global coordinates

GRID3_LEVEL = 0  # Finest grid refinement level
GRID3_NX = 8 * D  # Finest grid width
GRID3_NY = 6 * D  # Finest grid height
GRID3_X0 = 8 * D  # Left boundary in global coordinates
GRID3_X1 = GRID3_X0 + GRID3_NX  # Right boundary in global coordinates
GRID3_Y0 = (GRID1_NY - GRID3_NY) // 2  # Bottom boundary in global coordinates
GRID3_Y1 = GRID3_Y0 + GRID3_NY  # Top boundary in global coordinates

GRID2_X0_LOCAL, GRID2_Y0_LOCAL = mg.coord_to_indices(GRID2_X0, GRID2_Y0, 0, 0, GRID1_LEVEL)
GRID2_X1_LOCAL, GRID2_Y1_LOCAL = mg.coord_to_indices(GRID2_X1, GRID2_Y1, 0, 0, GRID1_LEVEL)
GRID3_X0_LOCAL, GRID3_Y0_LOCAL = mg.coord_to_indices(
    GRID3_X0, GRID3_Y0, GRID2_X0, GRID2_Y0, GRID2_LEVEL
)
GRID3_X1_LOCAL, GRID3_Y1_LOCAL = mg.coord_to_indices(
    GRID3_X1, GRID3_Y1, GRID2_X0, GRID2_Y0, GRID2_LEVEL
)
CYL_X_LOCAL, CYL_Y_LOCAL = mg.coord_to_indices(CYL_X, CYL_Y, GRID3_X0, GRID3_Y0, GRID3_LEVEL)

MARKER_THETA = jnp.linspace(0, 2 * jnp.pi, N_MARKER, dtype=jnp.float32, endpoint=False)
MARKER_X = CYL_X_LOCAL + 0.5 * D * jnp.cos(MARKER_THETA)  # Marker x-coordinates in Grid 3
MARKER_Y = CYL_Y_LOCAL + 0.5 * D * jnp.sin(MARKER_THETA)  # Marker y-coordinates in Grid 3
MARKER_COORDS = jnp.stack((MARKER_X, MARKER_Y), axis=1)
MARKER_DS = ib.get_ds_closed(MARKER_COORDS)  # Marker segment length


# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.05  # Inlet velocity
RE = 200  # Reynolds number
UR = 5  # Reduced velocity
MR = 10  # Mass ratio
DR = 0  # Damping ratio

NU = U0 * D / RE  # Kinematic viscosity
FN = U0 / (UR * D)  # Natural frequency of the structure
M = CYL_AREA * MR  # Mass of the cylinder per unit length
K = (2 * math.pi * FN) ** 2 * M  # Spring stiffness per unit length
C = 2 * math.sqrt(K * M) * DR  # Damping coefficient per unit length
TM = int(100 / FN / 4)  # Total simulation timesteps


# ========================== IB-LBM PARAMETERS ==========================

OMEGA1 = mg.get_omega(NU, GRID1_LEVEL)  # Grid 1 relaxation parameter
OMEGA2 = mg.get_omega(NU, GRID2_LEVEL)  # Grid 2 relaxation parameter
OMEGA3 = mg.get_omega(NU, GRID3_LEVEL)  # Grid 3 relaxation parameter

IB_PAD = 10  # Padding around cylinder for defining the local IBM region
IB_X0 = int(CYL_X_LOCAL - 0.5 * D - IB_PAD)  # X-coordinate of the initial IBM region
IB_Y0 = int(CYL_Y_LOCAL - 0.5 * D - IB_PAD)  # Y-coordinate of the initial IBM region
IB_SIZE = D + 2 * IB_PAD  # Size of the IBM region

IB_ITER = 1  # Multi-direct forcing iterations for IBM coupling
FSI_ITER = 1  # Number of fluid-structure iterations per timestep


# ======================= INITIALIZE VARIABLES ====================

f1, rho1, u1 = mg.init_grid(GRID1_NX, GRID1_NY, GRID1_LEVEL)
f2, rho2, u2 = mg.init_grid(GRID2_NX, GRID2_NY, GRID2_LEVEL)
f3, rho3, u3 = mg.init_grid(GRID3_NX, GRID3_NY, GRID3_LEVEL)

u1 = u1.at[0].set(U0)
u2 = u2.at[0].set(U0)
u3 = u3.at[0].set(U0)

f1 = lbm.get_equilibrium(rho1, u1)
f2 = lbm.get_equilibrium(rho2, u2)
f3 = lbm.get_equilibrium(rho3, u3)

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


def update_grid3(f2, f3, d, v, a):

    # Collision on the finest grid
    rho3, u3 = lbm.get_macroscopic(f3)
    feq3 = lbm.get_equilibrium(rho3, u3)
    f3 = lbm.collision_kbc(f3, feq3, OMEGA3)

    # Fluid-structure coupling on the finest grid
    f3, u3, d, v, a, h = solve_fsi(f3, rho3, u3, d, v, a)

    # Streaming and interpolation from Grid 2
    f3 = lbm.streaming(f3)
    f3 = mg.coarse_to_fine(f2[:, GRID3_X0_LOCAL, None, GRID3_Y0_LOCAL:GRID3_Y1_LOCAL], f3, dir="right")
    f3 = mg.coarse_to_fine(
        f2[:, GRID3_X1_LOCAL - 1, None, GRID3_Y0_LOCAL:GRID3_Y1_LOCAL],
        f3,
        dir="left",
    )
    f3 = mg.coarse_to_fine(f2[:, GRID3_X0_LOCAL:GRID3_X1_LOCAL, GRID3_Y1_LOCAL - 1, None], f3, dir="down")
    f3 = mg.coarse_to_fine(f2[:, GRID3_X0_LOCAL:GRID3_X1_LOCAL, GRID3_Y0_LOCAL, None], f3, dir="up")

    return f3, rho3, u3, d, v, a, h


def update_grid2(f1, f2, f3, d, v, a):

    # Collision on the intermediate grid
    rho2, u2 = lbm.get_macroscopic(f2)
    feq2 = lbm.get_equilibrium(rho2, u2)
    f2 = lbm.collision_kbc(f2, feq2, OMEGA2)

    # Streaming and interpolation from Grid 1
    f2 = lbm.streaming(f2)
    f2 = mg.coarse_to_fine(f1[:, GRID2_X0_LOCAL, None, GRID2_Y0_LOCAL:GRID2_Y1_LOCAL], f2, dir="right")
    f2 = mg.coarse_to_fine(
        f1[:, GRID2_X1_LOCAL - 1, None, GRID2_Y0_LOCAL:GRID2_Y1_LOCAL],
        f2,
        dir="left",
    )
    f2 = mg.coarse_to_fine(f1[:, GRID2_X0_LOCAL:GRID2_X1_LOCAL, GRID2_Y1_LOCAL - 1, None], f2, dir="down")
    f2 = mg.coarse_to_fine(f1[:, GRID2_X0_LOCAL:GRID2_X1_LOCAL, GRID2_Y0_LOCAL, None], f2, dir="up")

    # Grid 3 advances twice per Grid 2 update
    f3, rho3, u3, d, v, a, h = update_grid3(f2, f3, d, v, a)
    f3, rho3, u3, d, v, a, h = update_grid3(f2, f3, d, v, a)

    # Inject fine-grid data back into Grid 2
    f2 = f2.at[:, GRID3_X0_LOCAL, None, GRID3_Y0_LOCAL:GRID3_Y1_LOCAL].set(
        mg.fine_to_coarse(f3, f2[:, GRID3_X0_LOCAL, None, GRID3_Y0_LOCAL:GRID3_Y1_LOCAL], dir="left")
    )
    f2 = f2.at[:, GRID3_X1_LOCAL - 1, None, GRID3_Y0_LOCAL:GRID3_Y1_LOCAL].set(
        mg.fine_to_coarse(
            f3,
            f2[:, GRID3_X1_LOCAL - 1, None, GRID3_Y0_LOCAL:GRID3_Y1_LOCAL],
            dir="right",
        )
    )
    f2 = f2.at[:, GRID3_X0_LOCAL:GRID3_X1_LOCAL, GRID3_Y1_LOCAL - 1, None].set(
        mg.fine_to_coarse(
            f3,
            f2[:, GRID3_X0_LOCAL:GRID3_X1_LOCAL, GRID3_Y1_LOCAL - 1, None],
            dir="up",
        )
    )
    f2 = f2.at[:, GRID3_X0_LOCAL:GRID3_X1_LOCAL, GRID3_Y0_LOCAL, None].set(
        mg.fine_to_coarse(
            f3,
            f2[:, GRID3_X0_LOCAL:GRID3_X1_LOCAL, GRID3_Y0_LOCAL, None],
            dir="down",
        )
    )

    return f2, rho2, u2, f3, rho3, u3, d, v, a, h


@jax.jit
def update_grid1(f1, f2, f3, d, v, a):

    # Collision on the coarsest grid
    rho1, u1 = lbm.get_macroscopic(f1)
    feq1 = lbm.get_equilibrium(rho1, u1)
    f1 = lbm.collision_kbc(f1, feq1, OMEGA1)

    f1 = lbm.streaming(f1)
    f1 = lbm.boundary_nee(f1, loc="left", ux_wall=U0)
    f1 = lbm.boundary_equilibrium(f1, loc="right", ux_wall=U0)

    # Grid 2 advances twice per Grid 1 update
    f2, rho2, u2, f3, rho3, u3, d, v, a, h = update_grid2(f1, f2, f3, d, v, a)
    f2, rho2, u2, f3, rho3, u3, d, v, a, h = update_grid2(f1, f2, f3, d, v, a)

    # Inject intermediate-grid data back into Grid 1
    f1 = f1.at[:, GRID2_X0_LOCAL, None, GRID2_Y0_LOCAL:GRID2_Y1_LOCAL].set(
        mg.fine_to_coarse(f2, f1[:, GRID2_X0_LOCAL, None, GRID2_Y0_LOCAL:GRID2_Y1_LOCAL], dir="left")
    )
    f1 = f1.at[:, GRID2_X1_LOCAL - 1, None, GRID2_Y0_LOCAL:GRID2_Y1_LOCAL].set(
        mg.fine_to_coarse(
            f2,
            f1[:, GRID2_X1_LOCAL - 1, None, GRID2_Y0_LOCAL:GRID2_Y1_LOCAL],
            dir="right",
        )
    )
    f1 = f1.at[:, GRID2_X0_LOCAL:GRID2_X1_LOCAL, GRID2_Y1_LOCAL - 1, None].set(
        mg.fine_to_coarse(
            f2,
            f1[:, GRID2_X0_LOCAL:GRID2_X1_LOCAL, GRID2_Y1_LOCAL - 1, None],
            dir="up",
        )
    )
    f1 = f1.at[:, GRID2_X0_LOCAL:GRID2_X1_LOCAL, GRID2_Y0_LOCAL, None].set(
        mg.fine_to_coarse(
            f2,
            f1[:, GRID2_X0_LOCAL:GRID2_X1_LOCAL, GRID2_Y0_LOCAL, None],
            dir="down",
        )
    )

    return f1, rho1, u1, f2, rho2, u2, f3, rho3, u3, d, v, a, h


@jax.jit
def get_vorticity_image(u, local_diameter):
    return post.calculate_vorticity_dimensionless(u, local_diameter, U0).T


# ======================= VISUALIZATION SETUP ====================

mpl.rcParams["figure.raise_window"] = False

d_hist = np.zeros((2, TM), dtype=np.float32)
h_hist = np.zeros((2, TM), dtype=np.float32)

if PLOT:
    plt.figure(figsize=(10, 4))

    for x in range(0, GRID1_NX // 4 + 1):
        plt.plot([x * 4 / D, x * 4 / D], [0, GRID1_NY / D], "k", linewidth=0.05)
    for y in range(0, GRID1_NY // 4 + 1):
        plt.plot([0, GRID1_NX / D], [y * 4 / D, y * 4 / D], "k", linewidth=0.05)
    plt.text(0.1, 0.1, "Grid 1", ha="left", va="bottom", fontsize=6, color="black")

    for x in range(GRID2_X0 // 2, GRID2_X1 // 2 + 1):
        plt.plot([x * 2 / D, x * 2 / D], [GRID2_Y0 / D, GRID2_Y1 / D], "k", linewidth=0.05)
    for y in range(GRID2_Y0 // 2, GRID2_Y1 // 2 + 1):
        plt.plot([GRID2_X0 / D, GRID2_X1 / D], [y * 2 / D, y * 2 / D], "k", linewidth=0.05)
    plt.text(GRID2_X0 / D + 0.1, GRID2_Y0 / D + 0.1, "Grid 2", ha="left", va="bottom", fontsize=6, color="black")

    for x in range(GRID3_X0, GRID3_X1 + 1):
        plt.plot([x / D, x / D], [GRID3_Y0 / D, GRID3_Y1 / D], "k", linewidth=0.05)
    for y in range(GRID3_Y0, GRID3_Y1 + 1):
        plt.plot([GRID3_X0 / D, GRID3_X1 / D], [y / D, y / D], "k", linewidth=0.05)
    plt.text(GRID3_X0 / D + 0.1, GRID3_Y0 / D + 0.1, "Grid 3", ha="left", va="bottom", fontsize=6, color="black")

    ib_plot_x = (GRID3_X0 + IB_X0 + d[0]).astype(jnp.int32)
    ib_plot_y = (GRID3_Y0 + IB_Y0 + d[1]).astype(jnp.int32)
    ib_region = plt.Rectangle(
        (ib_plot_x / D, ib_plot_y / D),
        IB_SIZE / D,
        IB_SIZE / D,
        edgecolor="blue",
        linewidth=0.5,
        fill=False,
    )
    plt.gca().add_patch(ib_region)
    ib_label = plt.text(
        ib_plot_x / D + 0.5,
        ib_plot_y / D + 10,
        "IB Region",
        ha="center",
        va="bottom",
        fontsize=6,
        color="blue",
    )

    kwargs = dict(cmap="seismic", aspect="equal", origin="lower", vmax=10, vmin=-10)
    im1 = plt.imshow(
        get_vorticity_image(u1, D / 4) * 0,
        extent=[0, GRID1_NX / D, 0, GRID1_NY / D],
        **kwargs,
    )
    im2 = plt.imshow(
        get_vorticity_image(u2, D / 2),
        extent=[GRID2_X0 / D, GRID2_X1 / D, GRID2_Y0 / D, GRID2_Y1 / D],
        **kwargs,
    )
    im3 = plt.imshow(
        get_vorticity_image(u3, D),
        extent=[GRID3_X0 / D, GRID3_X1 / D, GRID3_Y0 / D, GRID3_Y1 / D],
        **kwargs,
    )

    plt.colorbar(label="Vorticity * D / U0")

    cylinder = plt.Circle(
        ((CYL_X + d[0]) / D, (CYL_Y + d[1]) / D),
        0.5,
        edgecolor="black",
        linewidth=0.5,
        facecolor="white",
        fill=True,
    )
    plt.gca().add_artist(cylinder)

    plt.plot(
        CYL_X / D,
        CYL_Y / D,
        marker="+",
        markersize=10,
        color="k",
        linestyle="None",
        markeredgewidth=0.5,
    )

    plt.xlabel("x / D")
    plt.ylabel("y / D")
    plt.tight_layout()


# ========================== RUN SIMULATION ==========================

def run_chunk(carry, n_steps):
    def step(carry, _):
        f1, u1, f2, u2, f3, u3, d, v, a = carry
        f1, _, u1, f2, _, u2, f3, _, u3, d, v, a, h = update_grid1(f1, f2, f3, d, v, a)
        return (f1, u1, f2, u2, f3, u3, d, v, a), (d, h)

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
        (f1, u1, f2, u2, f3, u3, d, v, a), (d_chunk, h_chunk) = run_chunk(
            (f1, u1, f2, u2, f3, u3, d, v, a),
            n_steps,
        )
        jax.block_until_ready(d_chunk)

        d_chunks.append(d_chunk)
        h_chunks.append(h_chunk)

        step_count += n_steps
        pbar.update(n_steps)

        if PLOT:
            im1.set_data(get_vorticity_image(u1, D / 4))
            im2.set_data(get_vorticity_image(u2, D / 2))
            im3.set_data(get_vorticity_image(u3, D))

            obj_x = (CYL_X + d[0]) / D
            obj_y = (CYL_Y + d[1]) / D
            cylinder.set_center((obj_x, obj_y))

            ib_region.set_xy(
                (
                    (GRID3_X0 + IB_X0 + d[0]).astype(jnp.int32) / D,
                    (GRID3_Y0 + IB_Y0 + d[1]).astype(jnp.int32) / D,
                )
            )
            ib_label.set_position((obj_x, obj_y + 0.6))

            plt.pause(0.01)

d_hist = jnp.swapaxes(jnp.concatenate(d_chunks, axis=0), 0, 1)
h_hist = jnp.swapaxes(jnp.concatenate(h_chunks, axis=0), 0, 1)

if PLOT:
    plt.show()
