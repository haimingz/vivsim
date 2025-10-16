"""
Vortex-Induced Vibration (VIV) of a circular cylinder using IB-LBM with Multi-GPU

This example simulates the vortex-induced vibration of a circular cylinder
in a 2D flow using the Immersed Boundary Method (IBM) coupled with the
Lattice Boltzmann Method (LBM). The cylinder is free to oscillate in both
cross-flow and in-line directions with spring-mass-damper constraints.

This implementation uses JAX's multi-device parallelization to distribute
the computation across multiple GPUs (or CPU cores for testing).

Note: Ensure that NY can be evenly divided by N_DEVICES.
"""

import math
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.shard_map import shard_map

from vivsim import dyn, ib, lbm, multidevice as md, post



# ========================== MULTI-GPU CONFIGURATION ====================

N_DEVICES = len(jax.devices())  # Number of GPU devices
mesh = Mesh(jax.devices(), axis_names=('y'))  # Divide domain along y-axis


# ========================== VISUALIZATION PARAMETERS ====================

PLOT = True                 # Enable real-time visualization
PLOT_EVERY = 100            # Plot update frequency
PLOT_AFTER = 0              # Start plotting after this timestep


# ========================== GEOMETRY =======================

D = 24                      # Cylinder diameter
NX = 20 * D                 # Domain width
NY = 10 * D                 # Domain height
X_CYL = 8 * D               # Cylinder center x-position
Y_CYL = 5 * D               # Cylinder center y-position
N_MARKER = 4 * D            # Number of markers on cylinder surface


# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.1                    # Inlet velocity
RE = 200                    # Reynolds number
UR = 5                      # Reduced velocity
MR = 10                     # Mass ratio
DR = 0                      # Damping ratio
NU = U0 * D / RE            # Kinematic viscosity
FN = U0 / (UR * D)                                  # Natural frequency
M = math.pi * (D / 2) ** 2 * MR                     # Mass of the cylinder
K = (FN * 2 * math.pi) ** 2 * M * (1 + 1 / MR)      # Spring stiffness
C = 2 * math.sqrt(K * M) * DR                       # Damping coefficient
TM = 60000                                          # Total simulation timesteps


# ========================== IB-LBM PARAMETERS ==========================

OMEGA = lbm.get_omega(NU)    # Relaxation parameter
IB_ITER = 3                  # Multi-direct forcing iterations
IB_LEFT = D                  # IBM region left padding
IB_RIGHT = 2 * D             # IBM region right padding


# ==================== MESHES ========================

# LBM grid coordinates
X, Y = jnp.meshgrid(jnp.arange(NX, dtype=jnp.int32), 
                    jnp.arange(NY, dtype=jnp.int32), 
                    indexing="ij")

# IBM marker configuration
THETA = jnp.linspace(0, 2 * jnp.pi, N_MARKER, endpoint=False)
X_MARKERS = X_CYL + 0.5 * D * jnp.cos(THETA)
Y_MARKERS = Y_CYL + 0.5 * D * jnp.sin(THETA)
DS_MARKERS = D * math.pi / N_MARKER

# Dynamic IBM region boundaries
IB_START_X = int(X_CYL - IB_LEFT)
IB_END_X = int(X_CYL + IB_RIGHT)


# ======================= INITIALIZE VARIABLES ====================

# Fluid variables
rho = jnp.ones((NX, NY), dtype=jnp.float32)
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)
u = u.at[0].set(U0)
f = lbm.get_equilibrium(rho, u)

# Structural variables
d = jnp.zeros(2, dtype=jnp.float32)            # Displacement of cylinder
v = jnp.zeros(2, dtype=jnp.float32)            # Velocity of cylinder
a = jnp.zeros(2, dtype=jnp.float32)            # Acceleration of cylinder
h = jnp.zeros(2, dtype=jnp.float32)            # Hydrodynamic force

# Optional: add initial perturbation
v = v.at[1].set(1e-2 * U0)


# ======================= SHARDING SPECIFICATIONS ====================

p_none = PartitionSpec(None)
p1 = PartitionSpec(None, 'y')
p2 = PartitionSpec(None, None, 'y')


# ======================= SIMULATION ROUTINE =====================

@jax.jit
@partial(
    shard_map, mesh=mesh,
    in_specs=(p2, p_none, p_none, p_none, p_none, p1, p1),
    out_specs=(p2, p1, p2, p_none, p_none, p_none, p_none)
)
def update(f, d, v, a, h, X, Y):
    
    # Update macroscopic variables
    rho, u = lbm.get_macroscopic(f)
    
    # Collision step
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)
    
    # Update markers position
    x_markers, y_markers = dyn.get_markers_coords_2dof(X_MARKERS, Y_MARKERS, d)
    
    # Extract data from IBM region for efficient computation
    u_ib = u[:, IB_START_X:IB_END_X]
    x_ib = X[IB_START_X:IB_END_X]
    y_ib = Y[IB_START_X:IB_END_X]
    f_ib = f[:, IB_START_X:IB_END_X]
    
    # Initialize forcing and marker forces
    g_ib = jnp.zeros((2, IB_END_X - IB_START_X, NY // N_DEVICES))
    h_marker = jnp.zeros((N_MARKER, 2))
    
    # Calculate kernel functions for all markers
    kernels = ib.get_kernels(x_markers, y_markers, x_ib, y_ib, ib.kernel_range4)
    
    # Multi-direct forcing iterations
    for _ in range(IB_ITER):
        
        # Velocity interpolation
        u_markers = jnp.einsum("dxy,nxy->nd", u_ib, kernels)
        u_markers = jax.lax.psum(u_markers, 'y')  # Multi-device synchronization
        
        # Compute correction force
        delta_u_markers = v - u_markers
        h_marker -= delta_u_markers * DS_MARKERS
        
        # Apply force to the fluid
        delta_u = jnp.einsum("nd,nxy->dxy", delta_u_markers, kernels) * DS_MARKERS
        g_ib += delta_u * 2
        u_ib += delta_u
    
    # Apply forcing to distribution functions (EDM forcing)
    rho_ib = jax.lax.dynamic_slice(rho, (IB_START_X, 0), (IB_END_X - IB_START_X, NY // N_DEVICES))
    f_ib = lbm.forcing_edm(f_ib, g_ib, u_ib, rho_ib)
    f = f.at[:, IB_START_X:IB_END_X].set(f_ib)
    
    # Update velocity in IBM region
    u_ib = u_ib + lbm.get_velocity_correction(g_ib, rho_ib)
    u = jax.lax.dynamic_update_slice(u, u_ib, (0, IB_START_X, 0))
    
    # Apply marker forces to the cylinder
    h = dyn.get_force_to_obj(h_marker)
    h += a * math.pi * D ** 2 / 4  # Internal mass correction
    a, v, d = dyn.newmark_2dof(a, v, d, h, M, K, C)
    
    # Streaming step
    f = lbm.streaming(f)
    f = md.stream_cross_devices(f, 'y', 'y', N_DEVICES)  # Multi-device streaming
    
    # Boundary conditions
    f = lbm.boundary_nebb(f, loc='left', ux_wall=U0)
    f = lbm.boundary_equilibrium(f, loc='right', ux_wall=U0)
    
    return f, rho, u, d, v, a, h


# ======================= VISUALIZATION SETUP ====================

mpl.rcParams['figure.raise_window'] = False

# Storage for displacement and force history
d_hist = np.zeros((2, TM))
h_hist = np.zeros((2, TM))

# Setup real-time visualization
if PLOT:
    plt.figure(figsize=(8, 3))
    
    curl = post.calculate_vorticity_dimensionless(u, D, U0)
    im = plt.imshow(
        curl.T,
        extent=[0, NX/D, 0, NY/D],
        cmap="bwr",
        aspect="equal",
        origin="lower",
        vmax=10, vmin=-10
    )
    
    plt.colorbar(label=r"$\omega D / U_0$")
    plt.xlabel(r"$x / D$")
    plt.ylabel(r"$y / D$")
    
    # Draw a circle representing the cylinder
    circle = plt.Circle(((X_CYL + d[0]) / D, (Y_CYL + d[1]) / D), 0.5, 
                        edgecolor='black', linewidth=0.5,
                        facecolor='white', fill=True)
    plt.gca().add_artist(circle)
    
    # Mark the initial position of the cylinder
    plt.plot((X_CYL + d[0]) / D, Y_CYL / D, marker='+', markersize=10, 
             color='k', linestyle='None', markeredgewidth=0.5)
    
    # Draw subdomain boundaries for multi-device visualization
    for i in range(N_DEVICES):
        plt.axhline(i * NY / N_DEVICES / D, color="r", linestyle="--", linewidth=0.5)
        plt.text(0.1, i * NY / N_DEVICES / D + 0.1, f'Device {i}', 
                 va='bottom', ha='left', fontsize=6, color='r', 
                 bbox=dict(facecolor='none', edgecolor='none', pad=1))
    
    plt.tight_layout()


# ========================== RUN SIMULATION ==========================

for t in tqdm(range(TM)):
    f, rho, u, d, v, a, h = update(f, d, v, a, h, X, Y)
    
    # Store history
    d_hist[:, t] = np.array(d)
    h_hist[:, t] = np.array(h)
    
    # Update visualization
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
        curl = post.calculate_vorticity_dimensionless(u, D, U0)
        im.set_data(curl.T)
        circle.center = ((X_CYL + d[0]) / D, (Y_CYL + d[1]) / D)
        plt.pause(0.001)
