# This example simulates the vortex-induced vibration of a circular cylinder
# in a 2D flow using the Immersed Boundary Method (IBM) coupled with the
# Lattice Boltzmann Method (LBM). The cylinder is placed at the center of the
# domain and is free to move in the flow. The flow is driven by a constant
# velocity U0 in the x direction. 
# 
# To test this script in a single-device environment, we fake 10 devices by 
# setting the environment variable `xla_force_host_platform_device_count=10`. 
# You can remove this line if you have multiple devices (e.g., GPUs). 
# But remember to make sure that NY can be evenly divided by N_DEVICES.

import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    # '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
    '--xla_force_host_platform_device_count=8'  # fake 8 devices, comment this if you do have multiple devices   
)
os.environ['JAX_PLATFORM_NAME'] = 'cpu' # use CPU cores to fake multiple devices, comment this if you do have multiple devices   

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


# ====================== plot options ======================

PLOT = True  # whether to plot the results
PLOT_EVERY = 100  # plot every n time steps
PLOT_AFTER = 00  # plot after n time steps


# ======================== multi-gpu parallelization ========================

N_DEVICES = len(jax.devices())  # number of gpu devices
mesh = Mesh(jax.devices(), axis_names=('y')) # divide the domain along the y-axis

# ======================== physical parameters =====================

RE = 100                                            # Reynolds number
UR = 5                                              # Reduced velocity
MR = 10                                             # Mass ratio
DR = 0                                              # Damping ratio
D = 30                                              # Cylinder diameter
U0 = 0.1                                            # Inlet velocity
TM = 60000                                          # Maximum number of time steps
NU = U0 * D / RE                                    # kinematic viscosity
FN = U0 / (UR * D)                                  # natural frequency
M = math.pi * (D / 2) ** 2 * MR                     # mass of the cylinder
K = (FN * 2 * math.pi) ** 2 * M * (1 + 1 / MR)      # stiffness
C = 2 * math.sqrt(K * M) * DR                       # damping

# =================== LBM parameters ==================

# LBM parameters
TAU = 3 * NU + 0.5  # relaxation time
OMEGA = 1 / TAU  # relaxation parameter

# MRT parameters
MRT_TRANS = mrt.get_trans_matrix()
MRT_RELAX = mrt.get_relax_matrix(OMEGA)
MRT_COL_LEFT = mrt.get_collision_left_matrix(MRT_TRANS, MRT_RELAX)  
MRT_SRC_LEFT = mrt.get_source_left_matrix(MRT_TRANS, MRT_RELAX)

# ================= fluid dynamics ==================

NX = 16 * D  # Number of grid points in x direction
NY = 8 * D   # Number of grid points in y direction

X, Y = jnp.meshgrid(jnp.arange(NX, dtype=jnp.uint16), 
                    jnp.arange(NY, dtype=jnp.uint16), 
                    indexing="ij")

rho = jnp.ones((NX, NY), dtype=jnp.float32)      # density of fluid
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)    # velocity of fluid
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)    # distribution functions
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # equilibrium distribution functions

# =================== dynamics of the cylinder ===================

d = jnp.zeros((2), dtype=jnp.float32)   # displacement of cylinder
v = jnp.zeros((2), dtype=jnp.float32)   # velocity of cylinder
a = jnp.zeros((2), dtype=jnp.float32)   # acceleration of cylinder
h = jnp.zeros((2), dtype=jnp.float32)   # hydrodynamic force


# =================== IB parameters ===================

N_MARKER = 4 * D                        # Number of markers on the circle
L_ARC = D * math.pi / N_MARKER          # arc length between the markers
N_ITER_MDF = 3                          # number of iterations for multi-direct forcing

X_OBJ = 6 * D                           # x-coordinate of the cylinder
Y_OBJ = NY / 2                          # y-coordinate of the cylinder

THETA_MAKERS = jnp.linspace(0, jnp.pi * 2, N_MARKER, dtype=jnp.float32, endpoint=False)
X_MARKERS = X_OBJ + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS = Y_OBJ + 0.5 * D * jnp.sin(THETA_MAKERS)

IBX1 = int(X_OBJ - 0.7 * D)             # left boundary of the IBM region 
IBX2 = int(X_OBJ + 1.0 * D)             # right boundary of the IBM region
# IBY1 = int(Y_OBJ - 1.5 * D)             # bottom boundary of the IBM region
# IBY2 = int(Y_OBJ + 1.5 * D)             # top boundary of the IBM region


# =================== initialize ===================

u = u.at[0].set(U0)
f = lbm.get_equilibrium(rho, u, f)
v = d.at[1].set(1e-2) # add an initial velocity to the cylinder


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

    # Immersed Boundary Method
    x_markers, y_markers = ib.update_markers_coords_2dof(X_MARKERS, Y_MARKERS, d)
    
    h_markers = jnp.zeros((x_markers.shape[0], 2))  # hydrodynamic force to the markers
    g = jnp.zeros((2, IBX2 - IBX1, NY // N_DEVICES))  # ! important for multi-device simulation
    
    # calculate the kernel functions for all markers
    kernels = ib.get_kernels(x_markers, y_markers, X[IBX1:IBX2], Y[IBX1:IBX2], ib.kernel_range4)
    
    for _ in range(N_ITER_MDF):
        
        # velocity interpolation
        u_markers = ib.interpolate_u_markers(u[:, IBX1:IBX2], kernels)
        u_markers = jax.lax.psum(u_markers, 'y')  # ! important for multi-device simulation
        
        # compute correction force
        g_markers_needed = ib.get_g_markers_needed(v, u_markers, L_ARC)
        g_needed = ib.spread_g_needed(g_markers_needed, kernels)
        
        # velocity correction
        u = u.at[:, IBX1:IBX2].add(lbm.get_velocity_correction(g_needed))
        
        # accumulate the corresponding correction force to the markers and the fluid
        h_markers -= g_markers_needed
        g += g_needed 

    # Compute force to the obj (including internal fluid force)
    h = ib.calculate_force_obj(h_markers)
    
    # eliminate internal fluid force (Feng's rigid body approximation)
    h -= a * math.pi * D ** 2 / 4 
    
    # Compute solid dynamics
    a, v, d = dyn.newmark_2dof(a, v, d, h, M, K, C)

    # Collision
    feq = lbm.get_equilibrium(rho, u, feq)
    f = mrt.collision(f, feq, MRT_COL_LEFT)
    
    # Add source term
    g_lattice = jnp.zeros((9, IBX2 - IBX1, NY // N_DEVICES))
    g_lattice = lbm.get_discretized_force(g, u[:, IBX1:IBX2],g_lattice)
    f = f.at[:, IBX1:IBX2].add(mrt.get_source(g_lattice, MRT_SRC_LEFT))  

    # Streaming
    f = lbm.streaming(f)
    f = md.stream_cross_devices(f, 'y', 'y', N_DEVICES) # ! important for multi-device simulation

    # Boundary conditions
    f = lbm.outlet_boundary(f, loc='right')
    f = lbm.velocity_boundary(f, U0, 0, loc='left')

    # update new macroscopic
    rho, u = lbm.get_macroscopic(f, rho, u)
     
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
    
    # draw outline of the IBM region
    plt.axvline(IBX1 / D, color="b", linestyle="--", linewidth=0.5)
    plt.axvline(IBX2 / D, color="b", linestyle="--", linewidth=0.5)
    plt.text((IBX1 + IBX2) / D / 2, NY / D - 0.2, 'IB region', color="b", fontsize=6, ha='center', va='top')
    
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
        # im.autoscale()
        circle.center = ((X_OBJ + d[0]) / D, (Y_OBJ + d[1]) / D)
        
        plt.pause(0.001)

