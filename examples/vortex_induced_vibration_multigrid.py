# This example simulates the vortex-induced vibration of a circular cylinder
# in a 2D flow using the Immersed Boundary Method (IBM) coupled with the
# Lattice Boltzmann Method (LBM). The cylinder is placed at the center of the
# domain and is free to move in the flow. The flow is driven by a constant
# velocity U0 in the x direction.

import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    # '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

import math
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from vivsim import dyn, ib, lbm, multigrid as mg, post, mrt


# ============================= plot options =======================

PLOT = False  # whether to plot the results
PLOT_EVERY = 100  # plot every n time steps
PLOT_AFTER = 00  # plot after n time steps


# ======================== physical parameters =====================

RE = 150                                        # Reynolds number
UR = 5                                          # Reduced velocity
MR = 10                                         # Mass ratio
DR = 0                                          # Damping ratio
U0 = 0.1                                        # Inlet velocity
D = 100                                          # Cylinder diameter
NU = U0 * D / RE                                # Kinematic viscosity
FN = U0 / (UR * D)                              # Natural frequency
TM = 30000                                      # Maximum time steps
M = math.pi * (D / 2) ** 2 * MR                 # Mass
K = (FN * 2 * math.pi) ** 2 * M * (1 + 1 / MR)  # Stiffness
C = 2 * math.sqrt(K * M) * DR                   # Damping


# ================= setup computation domain ==================

# ----------------- mesh #1 (coarse level = 1) -----------------

NX1 = 5 * D // 2
NY1 = 10 * D // 2

rho1 = jnp.ones((NX1 + 1, NY1), dtype=jnp.float32) 
u1 = jnp.zeros((2, NX1 + 1, NY1), dtype=jnp.float32)
f1 = jnp.zeros((9, NX1 + 1, NY1), dtype=jnp.float32)
feq1 = jnp.zeros((9, NX1 + 1, NY1), dtype=jnp.float32) 

# ----------------- mesh #2 (coarse level = 0) -----------------

NX2 = 8 * D 
NY2 = 10 * D

X2, Y2 = jnp.meshgrid(jnp.arange(NX2, dtype=jnp.uint16), jnp.arange(NY2, dtype=jnp.uint16), indexing="ij")

rho2 = jnp.ones((NX2, NY2), dtype=jnp.float32)
u2 = jnp.zeros((2, NX2, NY2), dtype=jnp.float32)
f2 = jnp.zeros((9, NX2, NY2), dtype=jnp.float32)
feq2 = jnp.zeros((9, NX2, NY2), dtype=jnp.float32)

# ----------------- mesh #3 (coarse level = 1) ----------------

NX3 = 3 * D // 2
NY3 = 10 * D // 2

rho3 = jnp.ones((NX3 + 1, NY3), dtype=jnp.float32)
u3 = jnp.zeros((2, NX3 + 1, NY3), dtype=jnp.float32)
f3 = jnp.zeros((9, NX3 + 1, NY3), dtype=jnp.float32)
feq3 = jnp.zeros((9, NX3 + 1, NY3), dtype=jnp.float32)

# ----------------- mesh #4 (coarse level = 2) ----------------

NX4 = 4 * D // 4
NY4 = 10 * D // 4

rho4 = jnp.ones((NX4 + 1, NY4), dtype=jnp.float32)
u4 = jnp.zeros((2, NX4 + 1, NY4), dtype=jnp.float32)
f4 = jnp.zeros((9, NX4 + 1, NY4), dtype=jnp.float32)
feq4 = jnp.zeros((9, NX4 + 1, NY4), dtype=jnp.float32)


# ===================== LBM parameters =====================

TAU = 3 * NU + 0.5 
MRT_TRANS = mrt.get_trans_matrix()

OMEGA1 = 2 / (TAU + 0.5)
MRT_RELAX1 = mrt.get_relax_matrix(OMEGA1)
MRT_COL_LEFT1 = mrt.get_collision_left_matrix(MRT_TRANS, MRT_RELAX1)  

OMEGA2 = 1 / TAU 
MRT_TRANS = mrt.get_trans_matrix()
MRT_RELAX2 = mrt.get_relax_matrix(OMEGA2)
MRT_COL_LEFT2 = mrt.get_collision_left_matrix(MRT_TRANS, MRT_RELAX2)  
MRT_SRC_LEFT2 = mrt.get_source_left_matrix(MRT_TRANS, MRT_RELAX2)

OMEGA3 = 2 / (TAU + 0.5)
MRT_RELAX3 = mrt.get_relax_matrix(OMEGA3)
MRT_COL_LEFT3 = mrt.get_collision_left_matrix(MRT_TRANS, MRT_RELAX3) 

OMEGA4 = 2 / (1 / OMEGA3 + 0.5)
MRT_RELAX4 = mrt.get_relax_matrix(OMEGA4)
MRT_COL_LEFT4 = mrt.get_collision_left_matrix(MRT_TRANS, MRT_RELAX4) 


# ====================== IBM parameters ==================

# location of the cylinder
X_OBJ = 2 * D   # x-coordinate of the cylinder
Y_OBJ = 5 * D   # y-coordinate of the cylinder

# Lagrangian markers
N_MARKER = 4 * D  # Number of markers on the circle
L_ARC = D * math.pi / N_MARKER  # arc length between the markers
THETA_MAKERS = jnp.linspace(0, jnp.pi * 2, N_MARKER, dtype=jnp.float32, endpoint=False)
X_MARKERS = X_OBJ + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS = Y_OBJ + 0.5 * D * jnp.sin(THETA_MAKERS)

# Multi-direct-forcing parameters
N_ITER_MDF = 5  # number of iterations for multi-direct forcing
IBX1 = int(X_OBJ - 0.7 * D)   # left boundary of the IBM region 
IBX2 = int(X_OBJ + 1.0 * D)   # right boundary of the IBM region
IBY1 = int(Y_OBJ - 1.5 * D)   # bottom boundary of the IBM region
IBY2 = int(Y_OBJ + 1.5 * D)   # top boundary of the IBM region

# dynamics
d = jnp.zeros((2), dtype=jnp.float32)  # displacement of cylinder
v = jnp.zeros((2), dtype=jnp.float32)  # velocity of cylinder
a = jnp.zeros((2), dtype=jnp.float32)  # acceleration of cylinder
h = jnp.zeros((2), dtype=jnp.float32)  # hydrodynamic force
forcing = jnp.zeros((9, IBX2 - IBX1, IBY2 - IBY1), dtype=jnp.float32)  # forcing term

# ======================= initialize =====================

u1 = u1.at[0].set(U0)
u2 = u2.at[0].set(U0)
u3 = u3.at[0].set(U0)
u4 = u4.at[0].set(U0)


f1 = lbm.get_equilibrium(rho1, u1, f1)
f2 = lbm.get_equilibrium(rho2, u2, f2)
f3 = lbm.get_equilibrium(rho3, u3, f3)
f4 = lbm.get_equilibrium(rho4, u4, f4)

v = v.at[1].set(1e-2) # add an initial velocity to the cylinder


# ======================= compute scheme =====================

def update_mesh2_locally(f, feq, rho, u, d, v, a, h):

    # Immersed Boundary Method
    x_markers, y_markers = ib.update_markers_coords_2dof(X_MARKERS, Y_MARKERS, d)
    
    h_markers = jnp.zeros((x_markers.shape[0], 2))  # hydrodynamic force to the markers
    g = jnp.zeros((2, IBX2 - IBX1, IBY2 - IBY1))  # distributed IB force to the fluid
       
    # calculate the kernel functions for all markers
    kernels = ib.get_kernels(x_markers, y_markers, X2[IBX1:IBX2, IBY1:IBY2], Y2[IBX1:IBX2, IBY1:IBY2], ib.kernel_range4)
    
    for _ in range(N_ITER_MDF):
        
        # velocity interpolation
        u_markers = ib.interpolate_u_markers(u[:, IBX1:IBX2, IBY1:IBY2], kernels)
        
        # compute correction force
        g_markers_needed = ib.get_g_markers_needed(v, u_markers, L_ARC)
        g_needed = ib.spread_g_needed(g_markers_needed, kernels)
        
        # velocity correction
        u = u.at[:, IBX1:IBX2, IBY1:IBY2].add(lbm.get_velocity_correction(g_needed))
        
        # accumulate the corresponding correction force to the markers and the fluid
        h_markers -= g_markers_needed
        g += g_needed 

    # Compute force to the obj (including internal fluid force)
    h = ib.calculate_force_obj(h_markers)
    
    # eliminate internal fluid force (Feng's rigid body approximation)
    h -= a * math.pi * D ** 2 / 4  # found unstable for high Re
    
    # Compute solid dynamics
    a, v, d = dyn.newmark_2dof(a, v, d, h, M, K, C)

    # Compute equilibrium
    feq = lbm.get_equilibrium(rho, u, feq)

    # Collision
    f = mrt.collision(f, feq, MRT_COL_LEFT2)
    
    # Add source term
    forcing = jnp.zeros((9, IBX2 - IBX1, IBY2 - IBY1), dtype=jnp.float32)
    forcing = lbm.get_discretized_force(g, u[:, IBX1:IBX2, IBY1:IBY2],forcing)
    f = f.at[:, IBX1:IBX2, IBY1:IBY2].add(mrt.get_source(forcing, MRT_SRC_LEFT2))
    
    return f, feq, rho, u, d, v, a, h

def update_mesh3_locally(f, feq, rho, u):    
    feq = lbm.get_equilibrium(rho, u, feq)
    f = mrt.collision(f, feq, MRT_COL_LEFT3)    
    return f, feq, rho, u

def update_mesh1_locally(f, feq, rho, u):    
    feq = lbm.get_equilibrium(rho, u, feq)
    f = mrt.collision(f, feq, MRT_COL_LEFT1)    
    return f, feq, rho, u

def update_mesh4_locally(f, feq, rho, u):    
    feq = lbm.get_equilibrium(rho, u, feq)
    f = mrt.collision(f, feq, MRT_COL_LEFT4)    
    return f, feq, rho, u

@jax.jit
def update(f1, f2, f3, f4, feq1, feq2, feq3, feq4, rho1, rho2, rho3, rho4, u1, u2, u3, u4, d, v, a, h):
    
    # collision (mesh4)
    rho4, u4 = lbm.get_macroscopic(f4, rho4, u4)
    f4, feq4, rho4, u4 = update_mesh1_locally(f4, feq4, rho4, u4)
    
    # reset ghost cells of mesh4
    f4 = mg.clear_ghost(f4, location='left')
    
    # collision (mesh1)
    rho1, u1 = lbm.get_macroscopic(f1, rho1, u1)
    f1, feq1, rho1, u1 = update_mesh1_locally(f1, feq1, rho1, u1)
    
    # collision (mesh3)
    rho3, u3 = lbm.get_macroscopic(f3, rho3, u3)
    f3, feq3, rho3, u3 = update_mesh3_locally(f3, feq3, rho3, u3)
    
    # reset ghost cells of mesh1 and mesh3
    f1 = mg.clear_ghost(f1, location='right')
    f3 = mg.clear_ghost(f3, location='left')
    
    # collision (mesh2)
    rho2, u2 = lbm.get_macroscopic(f2, rho2, u2)
    f2, feq2, rho2, u2, d, v, a, h = update_mesh2_locally(f2, feq2, rho2, u2, d, v, a, h)

    # streaming (mesh2)
    f1 = mg.accumulate(f2, f1, dir='left')
    f3 = mg.accumulate(f2, f3, dir='right')
    f2 = lbm.streaming(f2)
    f2, f3 = mg.explosion(f2, f3, dir='left')
    f2, f1 = mg.explosion(f2, f1, dir='right')  
    
    # collision (mesh2)
    rho2, u2 = lbm.get_macroscopic(f2, rho2, u2)
    f2, feq2, rho2, u2, d, v, a, h = update_mesh2_locally(f2, feq2, rho2, u2, d, v, a, h)
    
    # streaming (mesh2)
    f1 = mg.accumulate(f2, f1, dir='left')
    f3 = mg.accumulate(f2, f3, dir='right')
    f2 = lbm.streaming(f2)
    f2, f3 = mg.explosion(f2, f3, dir='left')
    f2, f1 = mg.explosion(f2, f1, dir='right')  
    
    # streaming (mesh1)
    f1 = f1.at[:,:-1].set(lbm.streaming(f1[:,:-1]))
    f1 = mg.coalescence(f1, dir='left')
    f1 = lbm.velocity_boundary(f1, U0, 0, loc='left')
    
    # streaming (mesh3)
    f4 = mg.accumulate(f3, f4, dir='right')
    f3 = f3.at[:, 1:].set(lbm.streaming(f3[:,1:]))
    f3 = mg.coalescence(f3, dir='right')
    f3, f4 = mg.explosion(f3, f4, dir='left')
    
    # collision (mesh1)
    rho1, u1 = lbm.get_macroscopic(f1, rho1, u1)
    f1, feq1, rho1, u1 = update_mesh1_locally(f1, feq1, rho1, u1)
    
    # collision (mesh3)
    rho3, u3 = lbm.get_macroscopic(f3, rho3, u3)
    f3, feq3, rho3, u3 = update_mesh3_locally(f3, feq3, rho3, u3)
    
    # reset ghost cells of mesh1 and mesh3
    f1 = mg.clear_ghost(f1, location='right')
    f3 = mg.clear_ghost(f3, location='left')
    
    # collision (mesh2)
    rho2, u2 = lbm.get_macroscopic(f2, rho2, u2)
    f2, feq2, rho2, u2, d, v, a, h = update_mesh2_locally(f2, feq2, rho2, u2, d, v, a, h)

    # streaming (mesh2)
    f1 = mg.accumulate(f2, f1, dir='left')
    f3 = mg.accumulate(f2, f3, dir='right')
    f2 = lbm.streaming(f2)
    f2, f3 = mg.explosion(f2, f3, dir='left')
    f2, f1 = mg.explosion(f2, f1, dir='right')  
    
    # collision (mesh2)
    rho2, u2 = lbm.get_macroscopic(f2, rho2, u2)
    f2, feq2, rho2, u2, d, v, a, h = update_mesh2_locally(f2, feq2, rho2, u2, d, v, a, h)
    
    # streaming (mesh2)
    f1 = mg.accumulate(f2, f1, dir='left')
    f3 = mg.accumulate(f2, f3, dir='right')
    f2 = lbm.streaming(f2)
    f2, f3 = mg.explosion(f2, f3, dir='left')
    f2, f1 = mg.explosion(f2, f1, dir='right')  
    
    # streaming (mesh1)
    f1 = f1.at[:,:-1].set(lbm.streaming(f1[:,:-1]))
    f1 = mg.coalescence(f1, dir='left')
    f1 = lbm.velocity_boundary(f1, U0, 0, loc='left')
    
    # streaming (mesh3)
    f4 = mg.accumulate(f3, f4, dir='right')
    f3 = f3.at[:, 1:].set(lbm.streaming(f3[:,1:]))
    f3 = mg.coalescence(f3, dir='right')
    f3, f4 = mg.explosion(f3, f4, dir='left')
    
    # streaming (mesh4)
    f4 = f4.at[:,1:].set(lbm.streaming(f4[:,1:]))
    f4 = mg.coalescence(f4, dir='right')   
    f4 = lbm.outlet_boundary(f4,loc='right') 
    
    return f1, f2, f3, f4, feq1, feq2, feq3, feq4, rho1, rho2, rho3, rho4, u1, u2, u3, u4, d, v, a, h


# ======================= create plot template =====================

if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    
    plt.figure(figsize=(8, 4))

    # curl = update_curl(curl, u1, u2, u3)
    vmax = 0.1

    im1 = plt.imshow(
        post.calculate_curl(u1[:,:-1]).T, 
        extent=[0, NX1 * 2 / D, 0, NY1 * 2 / D],
        cmap="seismic", aspect="equal", origin="lower",
        # norm=mpl.colors.CenteredNorm(),
        vmax=vmax,
        vmin=-1 * vmax,
    )
    
    im2 = plt.imshow(
        post.calculate_curl(u2).T * 2, 
        extent=[NX1 * 2 / D, (NX1 * 2 + NX2) / D, 0, NY2 / D],
        cmap="seismic", aspect="equal", origin="lower",
        # norm=mpl.colors.CenteredNorm(),
        vmax=vmax,
        vmin=-1 * vmax,
    )
    
    im3 = plt.imshow(
        post.calculate_curl(u3[:,1:]).T, 
        extent=[(NX1 * 2 + NX2) / D,(NX1 * 2 + NX2 + NX3 * 2) / D, 0, NY2 / D],
        cmap="seismic", aspect="equal", origin="lower",
        # norm=mpl.colors.CenteredNorm(),
        vmax=vmax,
        vmin=-1 * vmax,
    )
    
    im4 = plt.imshow(
        post.calculate_curl(u4[:,1:]).T,
        extent=[(NX1 * 2 + NX2 + NX3 * 2) / D, (NX1 * 2 + NX2 + NX3 * 2 + NX4 * 4) / D, 0, NY2 / D],
        cmap="seismic", aspect="equal", origin="lower",
        # norm=mpl.colors.CenteredNorm(),
        vmax=vmax,
        vmin=-1 * vmax,
    )

    plt.colorbar()
    plt.xlabel("x/D")
    plt.ylabel("y/D")

    # draw a circle representing the cylinder
    # circle = plt.Circle(((NX1 + X_OBJ + d[0]) / D , (Y_OBJ + d[1]) / D / 2), 0.25, 
                        # edgecolor='black', linewidth=0.5,
                        # facecolor='white', fill=True)
    
    # plt.gca().add_artist(circle)
    
    plt.axvline(NX1 * 2/ D, color="k", linestyle="--", linewidth=0.5)
    plt.axvline((NX1 * 2 + NX2) / D, color="k", linestyle="--", linewidth=0.5)
    plt.axvline((NX1 * 2 + NX2 + NX3 * 2) / D, color="k", linestyle="--", linewidth=0.5)
    
    plt.tight_layout()


# ========================== start simulation ==========================

for t in tqdm(range(TM)):
    f1, f2, f3, f4, feq1, feq2, feq3, feq4, rho1, rho2, rho3, rho4, u1, u2, u3, u4, d, v, a, h = update(
        f1, f2, f3, f4, feq1, feq2, feq3, feq4, rho1, rho2, rho3, rho4, u1, u2, u3, u4, d, v, a, h
    )

    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:

        im1.set_data(post.calculate_curl(u1[:,:-1]).T * 2)
        im2.set_data(post.calculate_curl(u2).T * 4)
        im3.set_data(post.calculate_curl(u3[:,1:]).T * 2)
        im4.set_data(post.calculate_curl(u4[:,1:]).T)
        
        # plt.gca().autoscale()
        
        # im1.autoscale()
        # im2.autoscale()
        # im3.autoscale()

        # circle.center = ((X_OBJ + d[0]) / D / 2, (Y_OBJ + d[1]) / D / 2)
        plt.pause(0.01)
