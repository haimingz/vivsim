# This example simulates the vortex-induced vibration of a circular cylinder
# in a 2D flow using the Immersed Boundary Method (IBM) coupled with the
# Lattice Boltzmann Method (LBM). The cylinder is placed at the center of the
# domain and is free to move in both directions with spring constraints. 
# The flow is driven by a constant velocity U0 in the x direction. 
# 
# To test this script in a single-device environment, we fake 8 devices by 
# setting the environment variable `xla_force_host_platform_device_count=8`. 
# You can remove this line if you have multiple devices (e.g., GPUs). 
# But remember to make sure that NY can be evenly divided by N_DEVICES.

import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    # '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
    # '--xla_force_host_platform_device_count=8'
)
# os.environ['JAX_PLATFORM_NAME'] = 'cpu' # use CPU cores to fake multiple devices  

import math
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from vivsim import dyn, ib, lbm, multidevice as md, post, mrt

from functools import partial
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.shard_map import shard_map

# ======================== multi-gpu parallelization ========================

N_DEVICES = len(jax.devices())  # number of gpu devices
mesh = Mesh(jax.devices(), axis_names=('y')) # divide the domain along the y-axis


# ============================= plot options =======================

PLOT = True
PLOT_EVERY = 100
PLOT_AFTER = 0

# ====================== Configuration ======================

# Simulation parameters
D = 24                 # Cylinder diameter
U0 = 0.1               # Inlet velocity
TM = 60000             # Total time steps

# Domain size
NX = 20 * D            # Grid points in x direction
NY = 10 * D            # Grid points in y direction

# Cylinder position
X_OBJ = 8 * D          # Cylinder x position
Y_OBJ = 5 * D          # Cylinder y position

# IB method parameters
N_MARKER = 4 * D       # Number of markers on cylinder
N_ITER_MDF = 3         # Multi-direct forcing iterations
IB_MARGIN = 2          # Margin of the IB region to the cylinder

# Physical parameters
RE = 200               # Reynolds number
UR = 5                 # Reduced velocity
MR = 10                # Mass ratio
DR = 0                 # Damping ratio

# =================== Pre-calculations ==================

# structural parameters
FN = U0 / (UR * D)                                          # Natural frequency
MASS = math.pi * (D / 2) ** 2 * MR                          # Mass of the cylinder
STIFFNESS = (FN * 2 * math.pi) ** 2 * MASS * (1 + 1 / MR)   # Stiffness of the spring
DAMPING = 2 * math.sqrt(STIFFNESS * MASS) * DR              # Damping of the spring

# fluid parameters
NU = U0 * D / RE                                            # Kinematic viscosity
TAU = 3 * NU + 0.5                                          # Relaxation time
OMEGA = 1 / TAU                                             # Relaxation parameter
MRT_COL_LEFT, MRT_SRC_LEFT = mrt.precompute_left_matrices(OMEGA)

# eulerian meshgrid
X, Y = jnp.meshgrid(jnp.arange(NX, dtype=jnp.int32), 
                    jnp.arange(NY, dtype=jnp.int32), 
                    indexing="ij")

# lagrangian markers
THETA_MAKERS = jnp.linspace(0, jnp.pi * 2, N_MARKER, dtype=jnp.float32, endpoint=False)
X_MARKERS = X_OBJ + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS = Y_OBJ + 0.5 * D * jnp.sin(THETA_MAKERS)
L_ARC = D * math.pi / N_MARKER

# dynamic ibm region
IB_START_X = int(X_OBJ - 0.5 * D - IB_MARGIN)
IB_START_Y = int(Y_OBJ - 0.5 * D - IB_MARGIN)
IB_SIZE = D + IB_MARGIN * 2

# =================== define variables ==================

# fluid variables
rho = jnp.ones((NX, NY), dtype=jnp.float32)      # density of fluid
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)    # velocity of fluid
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)    # distribution functions
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # equilibrium distribution functions

# structural variables
d = jnp.zeros((2), dtype=jnp.float32)   # displacement of cylinder
v = jnp.zeros((2), dtype=jnp.float32)   # velocity of cylinder
a = jnp.zeros((2), dtype=jnp.float32)   # acceleration of cylinder
h = jnp.zeros((2), dtype=jnp.float32)   # hydrodynamic force

# initial conditions
u = u.at[0].set(U0)
f = lbm.get_equilibrium(rho, u, f)
v = d.at[1].set(1e-2)  # add an initial velocity to the cylinder
feq_init = f[:,0,0]


# =================== define calculation routine ===================

p_none = PartitionSpec(None)
p1 = PartitionSpec(None, 'y')
p2 = PartitionSpec(None, None, 'y')

@jax.jit
@partial(
    shard_map, mesh=mesh,
    in_specs=(p2, p2, p1, p2, p_none, p_none, p_none, p_none, p1, p1),
    out_specs=(p2, p2, p1, p2, p_none, p_none, p_none, p_none)
)
def update(f, feq, rho, u, d, v, a, h, X, Y):
    
    # update new macroscopic
    rho, u = lbm.get_macroscopic(f, rho, u)
    
    # Collision
    feq = lbm.get_equilibrium(rho, u, feq)
    f = mrt.collision(f, feq, MRT_COL_LEFT)
    
    # update markers position
    x_markers, y_markers = ib.get_markers_coords_2dof(X_MARKERS, Y_MARKERS, d)
    
    # update ibm region
    ib_start_x = (IB_START_X + d[0]).astype(jnp.int32)
    ib_start_y = (IB_START_Y + d[1]).astype(jnp.int32)
    
    # extract data from ibm region
    u_slice = jax.lax.dynamic_slice(u, (0, ib_start_x, ib_start_y), (2, IB_SIZE, IB_SIZE))
    X_slice = jax.lax.dynamic_slice(X, (ib_start_x, ib_start_y), (IB_SIZE, IB_SIZE))
    Y_slice = jax.lax.dynamic_slice(Y, (ib_start_x, ib_start_y), (IB_SIZE, IB_SIZE))
    f_slice = jax.lax.dynamic_slice(f, (0, ib_start_x, ib_start_y), (9, IB_SIZE, IB_SIZE))
        
    g = jnp.zeros((2, IB_SIZE, IB_SIZE))  # ! important for multi-device simulation
    h_markers = jnp.zeros((N_MARKER, 2))  # hydrodynamic force to the markers
    
    # calculate the kernel functions for all markers
    kernels = ib.get_kernels(x_markers, y_markers, X_slice, Y_slice, ib.kernel_range4)
    
    for _ in range(N_ITER_MDF):
        
        # velocity interpolation
        u_markers = ib.interpolate_velocity_at_markers(u_slice, kernels)
        u_markers = jax.lax.psum(u_markers, 'y')  # ! important for multi-device simulation
        
        # compute correction force
        g_markers_correction = ib.get_noslip_forces_at_markers(v, u_markers, L_ARC)
        g_correction = ib.spread_force_to_fluid(g_markers_correction, kernels)
        
        # velocity correction
        u_slice += lbm.get_velocity_correction(g_correction)
        
        # accumulate the corresponding correction force to the markers and the fluid
        g += g_correction
        h_markers -= g_markers_correction

    g_lattice = lbm.get_discretized_force(g, u_slice)
    
    # apply the force to the lattice
    s_slice = mrt.get_source(g_lattice, MRT_SRC_LEFT)    
    f = jax.lax.dynamic_update_slice(f, f_slice + s_slice, (0, ib_start_x, ib_start_y))

    # apply the force to the cylinder
    h = ib.get_force_to_obj(h_markers)
    h += a * math.pi * D ** 2 / 4   
    a, v, d = dyn.newmark_2dof(a, v, d, h, MASS, STIFFNESS, DAMPING)
    
    # Streaming
    f = lbm.streaming(f)
    f = md.stream_cross_devices(f, 'y', 'y', N_DEVICES) # ! important for multi-device simulation

    # Boundary conditions
    f = lbm.boundary_equilibrium(f, feq_init[:,jnp.newaxis], loc='right')
    f = lbm.velocity_boundary(f, U0, 0, loc='left')

    return f, feq, rho, u, d, v, a, h


# =============== create plot template ================

if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    
    plt.figure(figsize=(8, 4))
    
    curl = post.calculate_curl(u)
    im = plt.imshow(
        curl.T,
        extent=[0, NX/D, 0, NY/D],
        cmap="seismic",
        aspect="equal",
        origin="lower",
        # norm=mpl.colors.CenteredNorm(),
        vmax=0.05,
        vmin=-0.05,
    )

    plt.colorbar()
    
    plt.xlabel("x/D")
    plt.ylabel("y/D")

    # draw a circle representing the cylinder
    circle = plt.Circle(((X_OBJ + d[0]) / D, (Y_OBJ + d[1]) / D), 0.5, 
                        edgecolor='black', linewidth=0.5,
                        facecolor='white', fill=True)
    plt.gca().add_artist(circle)
                
    # mark the initial position of the cylinder
    plt.plot((X_OBJ + d[0]) / D, Y_OBJ / D, marker='+', markersize=10, color='k', linestyle='None', markeredgewidth=0.5)
        
    # draw the boundary of subdomains corresponding to each device
    for i in range(N_DEVICES):
        plt.axhline(i * NY / N_DEVICES / D, color="r", linestyle="--", linewidth=0.5)
        plt.text(0.1, i * NY / N_DEVICES / D + 0.1, f'Device {i}', 
                 va='bottom', ha='left', fontsize=6, color='r', 
                 bbox=dict(facecolor='none', edgecolor='none', pad=1))
    
    plt.tight_layout()


# =============== start simulation ===============

for t in tqdm(range(TM)):
    f, feq, rho, u, d, v, a, h = update(f, feq, rho, u, d, v, a, h, X, Y)
    
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
        im.set_data(post.calculate_curl(u).T)
        circle.center = ((X_OBJ + d[0]) / D, (Y_OBJ + d[1]) / D)
        plt.pause(0.001)

