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

PLOT = True  # whether to plot the results
PLOT_EVERY = 100  # plot every n time steps
PLOT_AFTER = 00  # plot after n time steps


# ======================== physical parameters =====================

RE = 150                                        # Reynolds number
UR = 5                                          # Reduced velocity
MR = 10                                         # Mass ratio
DR = 0                                          # Damping ratio
U0 = 0.1                                        # Inlet velocity
D = 50                                         # Cylinder diameter
NU = U0 * D / RE                                # Kinematic viscosity
FN = U0 / (UR * D)                              # Natural frequency
TM = 30000                                      # Maximum time steps
M = math.pi * (D / 2) ** 2 * MR                 # Mass
K = (FN * 2 * math.pi) ** 2 * M * (1 + 1 / MR)  # Stiffness
C = 2 * math.sqrt(K * M) * DR                   # Damping


# ================= setup computation domain ==================

# ----------------- mesh #1 (coarse level = 1) -----------------

WIDTH1 = 5 * D 
HEIGHT1 = 12 * D

NX1 = WIDTH1 // 2 + 1
NY1 = HEIGHT1 // 2

rho1 = jnp.ones((NX1, NY1), dtype=jnp.float32) 
u1 = jnp.zeros((2, NX1, NY1), dtype=jnp.float32)
f1 = jnp.zeros((9, NX1, NY1), dtype=jnp.float32)
feq1 = jnp.zeros((9, NX1, NY1), dtype=jnp.float32) 

# ----------------- mesh #2 (coarse level = 0) -----------------

WIDTH2 = 8 * D
HEIGHT2 = 12 * D

NX2 = WIDTH2
NY2 = HEIGHT2

X2, Y2 = jnp.meshgrid(jnp.arange(NX2, dtype=jnp.uint16), jnp.arange(NY2, dtype=jnp.uint16), indexing="ij")

rho2 = jnp.ones((NX2, NY2), dtype=jnp.float32)
u2 = jnp.zeros((2, NX2, NY2), dtype=jnp.float32)
f2 = jnp.zeros((9, NX2, NY2), dtype=jnp.float32)
feq2 = jnp.zeros((9, NX2, NY2), dtype=jnp.float32)

# ----------------- mesh #3 (coarse level = 1) ----------------

WIDTH3 = 5 * D 
HEIGHT3 = 12 * D

NX3 = WIDTH3 // 2 + 1
NY3 = HEIGHT3 // 2

rho3 = jnp.ones((NX3, NY3), dtype=jnp.float32)
u3 = jnp.zeros((2, NX3 , NY3), dtype=jnp.float32)
f3 = jnp.zeros((9, NX3, NY3), dtype=jnp.float32)
feq3 = jnp.zeros((9, NX3, NY3), dtype=jnp.float32)

# ----------------- mesh #4 (coarse level = 2) ----------------

WIDTH4 = 5 * D 
HEIGHT4 = 12 * D

NX4 = WIDTH4 // 4 + 1
NY4 = HEIGHT4 // 4

rho4 = jnp.ones((NX4, NY4), dtype=jnp.float32)
u4 = jnp.zeros((2, NX4, NY4), dtype=jnp.float32)
f4 = jnp.zeros((9, NX4, NY4), dtype=jnp.float32)
feq4 = jnp.zeros((9, NX4, NY4), dtype=jnp.float32)


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
Y_OBJ = HEIGHT2 // 2   # y-coordinate of the cylinder

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


# ======================= initialize =====================

u1 = u1.at[0].set(U0)
u2 = u2.at[0].set(U0)
u3 = u3.at[0].set(U0)
u4 = u4.at[0].set(U0)


f1 = lbm.get_equilibrium(rho1, u1, f1)
f2 = lbm.get_equilibrium(rho2, u2, f2)
f3 = lbm.get_equilibrium(rho3, u3, f3)
f4 = lbm.get_equilibrium(rho4, u4, f4)
F4_INIT = f4

v = v.at[1].set(1e-3) # add an initial velocity to the cylinder


# ======================= compute routine =====================

def collision_mesh1(f, feq, rho, u):    
    rho, u = lbm.get_macroscopic(f, rho, u)
    feq = lbm.get_equilibrium(rho, u, feq)
    f = mrt.collision(f, feq, MRT_COL_LEFT1)    
    return f, feq, rho, u

def collision_mesh2(f, feq, rho, u, d, v, a, h):
    
    # LBM collision
    rho, u = lbm.get_macroscopic(f, rho, u)
    feq = lbm.get_equilibrium(rho, u, feq)
    f = mrt.collision(f, feq, MRT_COL_LEFT2)
      
    # Immersed Boundary Method
    ib_region = (slice(IBX1, IBX2), slice(IBY1, IBY2))
    x_markers, y_markers = ib.get_markers_coords_2dof(X_MARKERS, Y_MARKERS, d)
    g, h_markers = ib.multi_direct_forcing(rho[ib_region], u[:, *ib_region], X2[ib_region], Y2[ib_region],
                                           v, x_markers, y_markers, 
                                           N_MARKER, L_ARC, N_ITER_MDF, ib.kernel_range4)
    
    # Dynamics of the cylinder
    h = ib.get_force_to_obj(h_markers)
    h -= a * math.pi * D ** 2 / 4   
    a, v, d = dyn.newmark_2dof(a, v, d, h, M, K, C)
    
    # Add source term
    g_lattice = lbm.get_discretized_force(g, u[:, *ib_region])
    f = f.at[:, *ib_region].add(mrt.get_source(g_lattice, MRT_SRC_LEFT2))
    
    return f, feq, rho, u, d, v, a, h

def collision_mesh3(f, feq, rho, u):    
    rho, u = lbm.get_macroscopic(f, rho, u)
    feq = lbm.get_equilibrium(rho, u, feq)
    f = mrt.collision(f, feq, MRT_COL_LEFT3)    
    return f, feq, rho, u

def collision_mesh4(f, feq, rho, u):    
    rho, u = lbm.get_macroscopic(f, rho, u)
    feq = lbm.get_equilibrium(rho, u, feq)
    f = mrt.collision(f, feq, MRT_COL_LEFT4)    
    return f, feq, rho, u


def stream_mesh1(f1):
    f1 = f1.at[:,:-1].set(lbm.streaming(f1[:,:-1]))
    f1 = mg.coalescence(f1, dir='left')
    f1 = lbm.velocity_boundary(f1, U0, 0, loc='left')
    return f1

def stream_mesh2(f1, f2, f3):
    f1 = mg.accumulate(f2, f1, dir='left')
    f3 = mg.accumulate(f2, f3, dir='right')
    f2 = lbm.streaming(f2)
    f2, f3 = mg.explosion(f2, f3, dir='left')
    f2, f1 = mg.explosion(f2, f1, dir='right')  
    return f1, f2, f3

def stream_mesh3(f3, f4):
    f4 = mg.accumulate(f3, f4, dir='right')
    f3 = f3.at[:, 1:].set(lbm.streaming(f3[:,1:]))
    f3 = mg.coalescence(f3, dir='right')
    f3, f4 = mg.explosion(f3, f4, dir='left')
    return f3, f4

def stream_mesh4(f4):
    f4 = f4.at[:,1:].set(lbm.streaming(f4[:,1:]))
    f4 = mg.coalescence(f4, dir='right')   
    f4 = lbm.outlet_boundary_equilibrium(f4, F4_INIT, loc='right') 
    return f4


def update_mesh2(f1, f2, f3, feq2, rho2, u2, d, v, a, h):
    f2, feq2, rho2, u2, d, v, a, h = collision_mesh2(f2, feq2, rho2, u2, d, v, a, h)
    f1, f2, f3 = stream_mesh2(f1, f2, f3)
    return f1, f2, f3, feq2, rho2, u2, d, v, a, h

def update_mesh123(f1, f2, f3, f4, feq1, feq2, feq3, rho1, rho2, rho3, u1, u2, u3, d, v, a, h):
    
    # collision (mesh1 & mesh3)    
    f1, feq1, rho1, u1 = collision_mesh1(f1, feq1, rho1, u1)
    f3, feq3, rho3, u3 = collision_mesh3(f3, feq3, rho3, u3)
    
    # reset ghost cells (mesh1 & mesh3)
    f1 = mg.clear_ghost(f1, location='right')
    f3 = mg.clear_ghost(f3, location='left')
    
    # update fine mesh twice (mesh2)
    f1, f2, f3, feq2, rho2, u2, d, v, a, h = update_mesh2(f1, f2, f3, feq2, rho2, u2, d, v, a, h)
    f1, f2, f3, feq2, rho2, u2, d, v, a, h = update_mesh2(f1, f2, f3, feq2, rho2, u2, d, v, a, h)
    
    # streaming (mesh1 & mesh3)
    f1 = stream_mesh1(f1)    
    f3, f4 = stream_mesh3(f3, f4)
        
    return f1, f2, f3, f4, feq1, feq2, feq3, rho1, rho2, rho3, u1, u2, u3, d, v, a, h

@jax.jit
def update(f1, f2, f3, f4, feq1, feq2, feq3, feq4, rho1, rho2, rho3, rho4, u1, u2, u3, u4, d, v, a, h):
    
    # collision (mesh4)
    f4, feq4, rho4, u4 = collision_mesh1(f4, feq4, rho4, u4)
    
    # reset ghost cells of mesh4
    f4 = mg.clear_ghost(f4, location='left')
    
    # update fine mesh twice (mesh1 & mesh2 & mesh3)
    f1, f2, f3, f4, feq1, feq2, feq3, rho1, rho2, rho3, u1, u2, u3, d, v, a, h = update_mesh123(f1, f2, f3, f4, feq1, feq2, feq3, rho1, rho2, rho3, u1, u2, u3, d, v, a, h)    
    f1, f2, f3, f4, feq1, feq2, feq3, rho1, rho2, rho3, u1, u2, u3, d, v, a, h = update_mesh123(f1, f2, f3, f4, feq1, feq2, feq3, rho1, rho2, rho3, u1, u2, u3, d, v, a, h)
    
    # streaming (mesh4)
    f4 = stream_mesh4(f4)
    
    return f1, f2, f3, f4, feq1, feq2, feq3, feq4, rho1, rho2, rho3, rho4, u1, u2, u3, u4, d, v, a, h


# ======================= create plot template =====================

if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    
    plt.figure(figsize=(10, 4))

    # curl = update_curl(curl, u1, u2, u3)
    vmax = 0.1

    im1 = plt.imshow(
        post.calculate_curl(u1[:,:-1]).T, 
        extent=[0, WIDTH1 / D, 0, HEIGHT1 / D],
        cmap="seismic", aspect="equal", origin="lower",
        # norm=mpl.colors.CenteredNorm(),
        vmax=vmax,
        vmin=-1 * vmax,
    )
    
    im2 = plt.imshow(
        post.calculate_curl(u2).T * 2, 
        extent=[WIDTH1 / D, (WIDTH1 + WIDTH2) / D, 0, HEIGHT2 / D],
        cmap="seismic", aspect="equal", origin="lower",
        # norm=mpl.colors.CenteredNorm(),
        vmax=vmax,
        vmin=-1 * vmax,
    )
    
    im3 = plt.imshow(
        post.calculate_curl(u3[:,1:]).T, 
        extent=[(WIDTH1 + WIDTH2) / D, (WIDTH1 + WIDTH2 + WIDTH3) / D, 0, HEIGHT3 / D],
        cmap="seismic", aspect="equal", origin="lower",
        # norm=mpl.colors.CenteredNorm(),
        vmax=vmax,
        vmin=-1 * vmax,
    )
    
    im4 = plt.imshow(
        post.calculate_curl(u4[:,1:]).T,
        extent=[ (WIDTH1 + WIDTH2 + WIDTH3) / D,  (WIDTH1 + WIDTH2 + WIDTH3 + WIDTH4) / D, 0, HEIGHT4 / D],
        cmap="seismic", aspect="equal", origin="lower",
        # norm=mpl.colors.CenteredNorm(),
        vmax=vmax,
        vmin=-1 * vmax,
    )

    plt.colorbar()
    plt.xlabel("x/D")
    plt.ylabel("y/D")

    # draw a circle representing the cylinder
    circle = plt.Circle(((WIDTH1 + X_OBJ + d[0]) / D , (Y_OBJ + d[1]) / D / 2), 0.5, 
                        edgecolor='black', linewidth=0.5,
                        facecolor='white', fill=True)
    
    plt.gca().add_artist(circle)
    
    # draw the boundaries of mesh blocks
    plt.axvline(WIDTH1 / D, color="g", linestyle="--", linewidth=0.5)
    plt.text(WIDTH1 / D / 2, HEIGHT1 / D + 0.5, 
             'Mesh 1', color="g", fontsize=8, ha='center')
    
    plt.axvline((WIDTH1 + WIDTH2) / D, color="g", linestyle="--", linewidth=0.5)
    plt.text((WIDTH1 + WIDTH2 / 2) / D, HEIGHT2 / D + 0.5, 
             'Mesh 2', color="g", fontsize=8, ha='center')
    
    plt.axvline((WIDTH1 + WIDTH2 + WIDTH3) / D, color="g", linestyle="--", linewidth=0.5)
    plt.text((WIDTH1 + WIDTH2 + WIDTH3 / 2) / D, HEIGHT3 / D + 0.5, 
             'Mesh 3', color="g", fontsize=8, ha='center')
    
    plt.text((WIDTH1 + WIDTH2 + WIDTH3 + WIDTH4 / 2) / D, HEIGHT4 / D + 0.5, 
             'Mesh 4', color="g", fontsize=8, ha='center')
    
    # mark the initial position of the cylinder
    plt.plot((WIDTH1 + X_OBJ + d[0]) / D, Y_OBJ / D, marker='+', markersize=10, color='k', linestyle='None', markeredgewidth=0.5)
    
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
            
        circle.center = ((WIDTH1 + X_OBJ + d[0]) / D, (Y_OBJ + d[1]) / D)
        plt.pause(0.01)
