"""
Vortex-Induced Vibration (VIV) of a circular cylinder using IB-LBM with Multi-GPU

This example simulates the vortex-induced vibration of a circular cylinder
in a 2D flow using the Immersed Boundary Method (IBM) coupled with the
Lattice Boltzmann Method (LBM). The computation is distributed across
multiple devices using JAX sharding along the y direction.

Note: `NY` must be evenly divisible by `N_DEVICES`.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jax.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec
from tqdm import tqdm

from vivsim import dyn, ib, lbm, multidevice as md, post


# ========================== MULTI-DEVICE PARAMETERS ====================

N_DEVICES = len(jax.devices())  # Number of available devices
MESH = Mesh(jax.devices(), axis_names=("y",))  # Shard the domain along y

P_NONE = PartitionSpec(None)
P_1 = PartitionSpec(None, "y")
P_2 = PartitionSpec(None, None, "y")


# ========================== VISUALIZATION PARAMETERS ====================

PLOT = True  # Enable chunked live plotting (host sync on plot updates)
CHUNK_STEPS = 100


# ========================== GEOMETRY =======================

D = 24  # Cylinder diameter
NX = 20 * D  # Domain width
NY = 10 * D  # Domain height
CYL_X = 8 * D  # Cylinder center x-position
CYL_Y = 5 * D  # Cylinder center y-position
CYL_AREA = math.pi * (D / 2) ** 2  # Cylinder cross-sectional area

N_MARKER = 4 * D  # Number of markers on cylinder surface
MARKER_THETA = jnp.linspace(0, 2 * jnp.pi, N_MARKER, endpoint=False)
MARKER_X = CYL_X + 0.5 * D * jnp.cos(MARKER_THETA)  # Marker x-coordinates
MARKER_Y = CYL_Y + 0.5 * D * jnp.sin(MARKER_THETA)  # Marker y-coordinates
MARKER_COORDS = jnp.stack((MARKER_X, MARKER_Y), axis=1)
MARKER_DS = ib.get_ds_closed(MARKER_COORDS)  # Marker segment length


# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.1  # Inlet velocity
RE = 200  # Reynolds number
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

OMEGA = lbm.get_omega(NU)  # Relaxation parameter for LBM

IB_ITER = 3  # Multi-direct forcing iterations for IBM coupling
IB_LEFT_PAD = D  # Padding to the left of the cylinder
IB_RIGHT_PAD = 2 * D  # Padding to the right of the cylinder

IB_X0 = int(CYL_X - IB_LEFT_PAD)  # Left boundary of the local IBM region
IB_X1 = int(CYL_X + IB_RIGHT_PAD)  # Right boundary of the local IBM region


# ========================== GRID COORDINATES =======================

X, Y = jnp.meshgrid(
    jnp.arange(NX, dtype=jnp.int32),
    jnp.arange(NY, dtype=jnp.int32),
    indexing="ij",
)


# ======================= INITIALIZE VARIABLES ====================

rho = jnp.ones((NX, NY), dtype=jnp.float32)  # Fluid density
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)  # Fluid velocity
u = u.at[0].set(U0)  # Set initial x-velocity to U0 for a uniform inlet flow
f = lbm.get_equilibrium(rho, u)  # Initial LBM distribution function

d = jnp.zeros(2, dtype=jnp.float32)  # Displacement of cylinder
v = jnp.zeros(2, dtype=jnp.float32)  # Velocity of cylinder
a = jnp.zeros(2, dtype=jnp.float32)  # Acceleration of cylinder
h = jnp.zeros(2, dtype=jnp.float32)  # Hydrodynamic force on the cylinder

# Optional: add a small cross-flow perturbation
v = v.at[1].set(1e-2 * U0)


# ======================= SIMULATION ROUTINE =====================

@jax.jit
@partial(
    shard_map,
    mesh=MESH,
    in_specs=(P_2, P_NONE, P_NONE, P_NONE, P_NONE, P_1, P_1),
    out_specs=(P_2, P_1, P_2, P_NONE, P_NONE, P_NONE, P_NONE),
)
def update(f, d, v, a, h, X, Y):

    # Collision
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)

    # Update marker positions based on the current displacement
    marker_x, marker_y = dyn.get_markers_coords_2dof(MARKER_X, MARKER_Y, d)

    # Extract IBM region data for efficient computation
    ib_u = u[:, IB_X0:IB_X1]
    ib_f = f[:, IB_X0:IB_X1]
    ib_rho = jax.lax.dynamic_slice(rho, (IB_X0, 0), (IB_X1 - IB_X0, NY // N_DEVICES))

    ib_g = jnp.zeros((2, IB_X1 - IB_X0, NY // N_DEVICES))
    marker_h = jnp.zeros((N_MARKER, 2))

    ib_x = X[IB_X0:IB_X1]
    ib_y = Y[IB_X0:IB_X1]
    kernels = jax.vmap(
        lambda xm, ym: ib.kernel_peskin_4pt(ib_x - xm) * ib.kernel_peskin_4pt(ib_y - ym)
    )(marker_x, marker_y)

    # Run multi-direct forcing across all devices
    for _ in range(IB_ITER):
        marker_u = jnp.einsum("dxy,nxy->nd", ib_u, kernels)
        marker_u = jax.lax.psum(marker_u, "y")

        delta_marker_u = v - marker_u
        marker_h -= delta_marker_u * MARKER_DS[:, None]

        delta_u = jnp.einsum("nd,nxy->dxy", delta_marker_u * MARKER_DS[:, None], kernels)
        ib_g += 2 * delta_u
        ib_u += delta_u

    # Apply IBM forcing to the fluid in the local IBM region
    ib_f = lbm.forcing_edm(ib_f, ib_g, ib_u, ib_rho)
    f = f.at[:, IB_X0:IB_X1].set(ib_f)

    ib_u = ib_u + lbm.get_velocity_correction(ib_g, ib_rho)
    u = jax.lax.dynamic_update_slice(u, ib_u, (0, IB_X0, 0))

    h = dyn.get_force_to_obj(marker_h)
    h += a * CYL_AREA
    a, v, d = dyn.newmark_2dof(a, v, d, h, M, K, C)

    # Streaming and boundary conditions
    f = lbm.streaming(f)
    f = md.stream_cross_devices(f, "y", "y", N_DEVICES)
    f = lbm.boundary_nebb(f, loc="left", ux_wall=U0)
    f = lbm.boundary_equilibrium(f, loc="right", ux_wall=U0)

    return f, rho, u, d, v, a, h


@jax.jit
def get_vorticity_image(u):
    return post.calculate_vorticity_dimensionless(u, D, U0).T


# ======================= VISUALIZATION SETUP ====================

mpl.rcParams["figure.raise_window"] = False

d_hist = np.zeros((2, TM), dtype=np.float32)
h_hist = np.zeros((2, TM), dtype=np.float32)

if PLOT:
    plt.figure(figsize=(8, 3))

    im = plt.imshow(
        get_vorticity_image(u),
        extent=[0, NX / D, 0, NY / D],
        cmap="bwr",
        aspect="equal",
        origin="lower",
        vmax=10,
        vmin=-10,
    )

    plt.colorbar(label="Vorticity * D / U0")
    plt.xlabel("x / D")
    plt.ylabel("y / D")

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

    for i in range(N_DEVICES):
        y0 = i * NY / N_DEVICES / D
        plt.axhline(y0, color="r", linestyle="--", linewidth=0.5)
        plt.text(
            0.1,
            y0 + 0.1,
            f"Device {i}",
            va="bottom",
            ha="left",
            fontsize=6,
            color="r",
            bbox=dict(facecolor="none", edgecolor="none", pad=1),
        )

    plt.tight_layout()


# ========================== RUN SIMULATION ==========================

def run_chunk(carry, n_steps):
    def step(carry, _):
        f, u, d, v, a, h = carry
        f, _, u, d, v, a, h = update(f, d, v, a, h, X, Y)
        return (f, u, d, v, a, h), (d, h)

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
        (f, u, d, v, a, h), (d_chunk, h_chunk) = run_chunk((f, u, d, v, a, h), n_steps)
        jax.block_until_ready(d_chunk)

        d_chunks.append(d_chunk)
        h_chunks.append(h_chunk)

        step_count += n_steps
        pbar.update(n_steps)

        if PLOT:
            im.set_data(get_vorticity_image(u))
            cylinder.center = ((CYL_X + d[0]) / D, (CYL_Y + d[1]) / D)
            plt.pause(0.001)

d_hist = jnp.swapaxes(jnp.concatenate(d_chunks, axis=0), 0, 1)
h_hist = jnp.swapaxes(jnp.concatenate(h_chunks, axis=0), 0, 1)

if PLOT:
    plt.show()
