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
    # '--xla_force_host_platform_device_count=12'  # fake 8 devices, comment this if you do have multiple devices   
)
# os.environ['JAX_PLATFORM_NAME'] = 'cpu' # use CPU cores to fake multiple devices, comment this if you do have multiple devices   

import math
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

from vivsim import dyn, ib, lbm, multigrid as mg, post, mrt, multidevice as md

# ============================= plot options =======================

PLOT = 'curl'  # whether to plot the results
PLOT_EVERY = 100  # plot every n time steps
PLOT_AFTER = 00  # plot after n time steps

# ====================== Configuration ======================

# LBM parameters
D = 24                 # Cylinder diameter
U0 = 0.1               # Inlet velocity
TM = 60000             # Total time steps

# multi-block config
HEIGHT = 10 * D
WIDTH = [7 * D, 3 * D, 1 * D, 9 * D]
LEVELS = [-1, 0, -1, -2]

# cylinder position
X_OBJ = 8 * D
Y_OBJ = HEIGHT // 2 

# IB parameters
N_MARKER = 4 * D
N_ITER_MDF = 3
IB_MARGIN = 3          # Margin of the IB region to the cylinder

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

MRT_COL_LEFT1, MRT_SRC_LEFT1 = mrt.precompute_left_matrices(mg.get_omega(NU, LEVELS[0]))
MRT_COL_LEFT2, MRT_SRC_LEFT2 = mrt.precompute_left_matrices(mg.get_omega(NU, LEVELS[1]))
MRT_COL_LEFT3, MRT_SRC_LEFT3 = mrt.precompute_left_matrices(mg.get_omega(NU, LEVELS[2]))
MRT_COL_LEFT4, MRT_SRC_LEFT4 = mrt.precompute_left_matrices(mg.get_omega(NU, LEVELS[3]))

# ====================== IBM parameters ==================

X_OBJ_LOCAL = X_OBJ - WIDTH[0]
Y_OBJ_LOCAL = Y_OBJ

THETA_MAKERS = jnp.linspace(0, jnp.pi * 2, N_MARKER, dtype=jnp.float32, endpoint=False)
X_MARKERS_LOCAL = X_OBJ_LOCAL + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS_LOCAL = Y_OBJ_LOCAL + 0.5 * D * jnp.sin(THETA_MAKERS)

L_ARC = D * math.pi / N_MARKER  # arc length between the markers

# dynamic ibm region
IB_START_X = int(X_OBJ_LOCAL - 0.5 * D - IB_MARGIN)
IB_START_Y = int(Y_OBJ_LOCAL - 0.5 * D - IB_MARGIN)
IB_SIZE = D + IB_MARGIN * 2
 
X, Y = jnp.meshgrid(jnp.arange(WIDTH[1]), jnp.arange(HEIGHT), indexing='ij')


# ======================= define variables =====================

f1, feq1, rho1, u1 = mg.generate_block_data(WIDTH[0], HEIGHT, LEVELS[0])
f2, feq2, rho2, u2 = mg.generate_block_data(WIDTH[1], HEIGHT, LEVELS[1])
f3, feq3, rho3, u3 = mg.generate_block_data(WIDTH[2], HEIGHT, LEVELS[2])
f4, feq4, rho4, u4 = mg.generate_block_data(WIDTH[3], HEIGHT, LEVELS[3])

u1 = u1.at[0].set(U0)
u2 = u2.at[0].set(U0)
u3 = u3.at[0].set(U0)
u4 = u4.at[0].set(U0)

f1 = lbm.get_equilibrium(rho1, u1, f1)
f2 = lbm.get_equilibrium(rho2, u2, f2)
f3 = lbm.get_equilibrium(rho3, u3, f3)
f4 = lbm.get_equilibrium(rho4, u4, f4)

d = jnp.zeros((2), dtype=jnp.float32) 
v = jnp.zeros((2), dtype=jnp.float32) 
a = jnp.zeros((2), dtype=jnp.float32) 
h = jnp.zeros((2), dtype=jnp.float32) 

v = v.at[1].set(1e-3) 

feq0 = f4[:,0,0]

# ======================= compute routine =====================


def macro_collision(f, feq, rho, u, left_matrix):  
    rho, u = lbm.get_macroscopic(f, rho, u)
    feq = lbm.get_equilibrium(rho, u, feq)
    f = mrt.collision(f, feq, left_matrix)
    return f, feq, rho, u

def solve_fsi(f, rho, u, d, v, a, h):
    
    # update markers position
    x_markers, y_markers = ib.get_markers_coords_2dof(X_MARKERS_LOCAL, Y_MARKERS_LOCAL, d)
    
    # update ibm region
    ib_start_x = (IB_START_X + d[0]).astype(jnp.int32)
    ib_start_y = (IB_START_Y + d[1]).astype(jnp.int32)
    
    # extract data from ibm region
    u_slice = jax.lax.dynamic_slice(u, (0, ib_start_x, ib_start_y), (2, IB_SIZE, IB_SIZE))
    X_slice = jax.lax.dynamic_slice(X, (ib_start_x, ib_start_y), (IB_SIZE, IB_SIZE))
    Y_slice = jax.lax.dynamic_slice(Y, (ib_start_x, ib_start_y), (IB_SIZE, IB_SIZE))
    f_slice = jax.lax.dynamic_slice(f, (0, ib_start_x, ib_start_y), (9, IB_SIZE, IB_SIZE))
    
    # calculate ibm force
    g_lattice, h_markers = ib.multi_direct_forcing(u_slice, X_slice, Y_slice, 
                                                   v, x_markers, y_markers, N_MARKER, L_ARC, 
                                                   N_ITER_MDF, ib.kernel_range4)

    # apply the force to the lattice
    s_slice = mrt.get_source(g_lattice, MRT_SRC_LEFT2)    
    f = jax.lax.dynamic_update_slice(f, f_slice + s_slice, (0, ib_start_x, ib_start_y))

    # apply the force to the cylinder
    h = ib.get_force_to_obj(h_markers)
    h += a * math.pi * D ** 2 / 4   
    a, v, d = dyn.newmark_2dof(a, v, d, h, MASS, STIFFNESS, DAMPING)
    
    return f, d, v, a, h

def update_coarse0(f1, f2, feq2, rho2, u2, f3, d, v, a, h):
    
    f2, feq2, rho2, u2 = macro_collision(f2, feq2, rho2, u2, MRT_COL_LEFT2)
    f2, d, v, a, h = solve_fsi(f2, rho2, u2, d, v, a, h)
    
    f1 = mg.accumulate(f2, f1, dir='left')
    f3 = mg.accumulate(f2, f3, dir='right')
    f2 = lbm.streaming(f2)
    f2, f1 = mg.explosion(f2, f1, dir='right')
    f2, f3 = mg.explosion(f2, f3, dir='left')
    
    return f1, f2, feq2, rho2, u2, f3, d, v, a, h

def update_coarse1(f1, feq1, rho1, u1, 
                   f2, feq2, rho2, u2,
                   f3, feq3, rho3, u3, 
                   f4, d, v, a, h):
    
    # collision (mesh1 & mesh3)
    f1, feq1, rho1, u1 = macro_collision(f1, feq1, rho1, u1, MRT_COL_LEFT1)
    f3, feq3, rho3, u3 = macro_collision(f3, feq3, rho3, u3, MRT_COL_LEFT3)
    
    # reset ghost cells (mesh1 & mesh3)
    f1 = mg.clear_ghost(f1, location='right')
    f3 = mg.clear_ghost(f3, location='left')
    
    # update fine mesh twice (mesh2)
    f1, f2, feq2, rho2, u2, f3, d, v, a, h = update_coarse0(f1, f2, feq2, rho2, u2, f3, d, v, a, h)
    f1, f2, feq2, rho2, u2, f3, d, v, a, h = update_coarse0(f1, f2, feq2, rho2, u2, f3, d, v, a, h)
    
    # streaming (mesh1 & mesh3)
    f1 = f1.at[:,:-1].set(lbm.streaming(f1[:,:-1]))
    f1 = mg.coalescence(f1, dir='left')
    f1 = lbm.velocity_boundary(f1, U0, 0, loc='left')
     
    f4 = mg.accumulate(f3, f4, dir='right')
    f3 = f3.at[:, 1:].set(lbm.streaming(f3[:,1:]))
    f3 = mg.coalescence(f3, dir='right')
    f3, f4 = mg.explosion(f3, f4, dir='left')
        
    return (f1, feq1, rho1, u1, 
            f2, feq2, rho2, u2,
            f3, feq3, rho3, u3, 
            f4, d, v, a, h,)

@jax.jit
def update_coarse2(f1, feq1, rho1, u1, 
                   f2, feq2, rho2, u2, 
                   f3, feq3, rho3, u3, 
                   f4, feq4, rho4, u4, 
                   d, v, a, h):
    
    # collision (mesh4)
    f4, feq4, rho4, u4 = macro_collision(f4, feq4, rho4, u4, MRT_COL_LEFT4)
    
    # reset ghost cells of mesh4
    f4 = mg.clear_ghost(f4, location='left')
    
    # update fine mesh twice (mesh1 & mesh2 & mesh3)
    (f1, feq1, rho1, u1, 
     f2, feq2, rho2, u2,
     f3, feq3, rho3, u3, 
     f4, d, v, a, h) = update_coarse1(f1, feq1, rho1, u1, 
                                       f2, feq2, rho2, u2,
                                       f3, feq3, rho3, u3, 
                                       f4, d, v, a, h)    
    (f1, feq1, rho1, u1, 
     f2, feq2, rho2, u2,
     f3, feq3, rho3, u3, 
     f4, d, v, a, h) = update_coarse1(f1, feq1, rho1, u1, 
                                       f2, feq2, rho2, u2,
                                       f3, feq3, rho3, u3, 
                                       f4, d, v, a, h)   

    # streaming (mesh4)
    f4 = f4.at[:,1:].set(lbm.streaming(f4[:,1:]))
    f4 = mg.coalescence(f4, dir='right')   
    f4 = lbm.boundary_equilibrium(f4, feq0[:, jnp.newaxis], loc='right')
    
    return (f1, feq1, rho1, u1, 
            f2, feq2, rho2, u2, 
            f3, feq3, rho3, u3, 
            f4, feq4, rho4, u4, 
            d, v, a, h)


# ======================= create plot template =====================

if PLOT == 'curl':
    mpl.rcParams['figure.raise_window'] = False
    
    plt.figure(figsize=(10, 4))

    kwargs = dict(
        cmap="seismic", aspect="equal", origin="lower",
        # norm=mpl.colors.CenteredNorm(),
        vmax=0.3,
        vmin=-0.3,
    )

    im1 = plt.imshow(
        post.calculate_curl(u1[:,:-1]).T, 
        extent=[0, WIDTH[0] / D, 0, HEIGHT / D],
        **kwargs
    )
    
    im2 = plt.imshow(
        post.calculate_curl(u2).T * 2, 
        extent=[WIDTH[0] / D, (WIDTH[0] + WIDTH[1]) / D, 0, HEIGHT / D],
        **kwargs
    )
    
    im3 = plt.imshow(
        post.calculate_curl(u3[:,1:]).T, 
        extent=[(WIDTH[0] + WIDTH[1]) / D, (WIDTH[0] + WIDTH[1] + WIDTH[2]) / D, 0, HEIGHT / D],
        **kwargs
    )
    
    im4 = plt.imshow(
        post.calculate_curl(u4[:,1:]).T,
        extent=[ (WIDTH[0] + WIDTH[1] + WIDTH[2]) / D,  (WIDTH[0] + WIDTH[1] + WIDTH[2] + WIDTH[3]) / D, 0, HEIGHT / D],
        **kwargs
    )

    plt.colorbar()
    plt.xlabel("x/D")
    plt.ylabel("y/D")

    # draw a circle representing the cylinder
    circle = plt.Circle(((X_OBJ + d[0]) / D , (Y_OBJ + d[1]) / D ), 0.5, 
                        edgecolor='black', linewidth=0.5,
                        facecolor='white', fill=True)
    plt.gca().add_artist(circle)
    
    # draw the boundaries of mesh blocks
    plt.axvline(WIDTH[0] / D, color="g", linestyle="--", linewidth=0.5)
    plt.axvline((WIDTH[0] + WIDTH[1]) / D, color="g", linestyle="--", linewidth=0.5)   
    plt.axvline((WIDTH[0] + WIDTH[1] + WIDTH[2]) / D, color="g", linestyle="--", linewidth=0.5)

    # mark the initial position of the cylinder
    plt.plot(X_OBJ / D, Y_OBJ / D, 
             marker='+', markersize=10, color='k', linestyle='None', markeredgewidth=0.5)
    
    plt.tight_layout()

dy = []

if PLOT == 'viv':
    mpl.rcParams['figure.raise_window'] = False
    plt.figure(figsize=(10, 4))    
    line, = plt.plot(dy, label='displacement')
    title = plt.title('')
    plt.ylim(-1, 1)
    

# ========================== start simulation ==========================

for t in tqdm(range(TM)):
    (
        f1, feq1, rho1, u1, 
        f2, feq2, rho2, u2, 
        f3, feq3, rho3, u3, 
        f4, feq4, rho4, u4, 
        d, v, a, h
    ) = update_coarse2(
        f1, feq1, rho1, u1, 
        f2, feq2, rho2, u2, 
        f3, feq3, rho3, u3, 
        f4, feq4, rho4, u4, 
        d, v, a, h,
    )
    
    if t % PLOT_EVERY == 0 and t > PLOT_AFTER:
    
        if PLOT == 'curl':
            im1.set_data(post.calculate_curl(u1[:,:-1]).T * 2)
            im2.set_data(post.calculate_curl(u2).T * 4)
            im3.set_data(post.calculate_curl(u3[:,1:]).T * 2)
            im4.set_data(post.calculate_curl(u4[:,1:]).T)
            circle.center = ((X_OBJ + d[0]) / D, (Y_OBJ + d[1]) / D)
            plt.pause(0.01)

        if PLOT == 'viv':            
            line.set_data(range(len(dy)),dy)
            title.set_text(f'amp: {max(dy[-1000:]):.3f}D')
            plt.xlim(0, len(dy))
            plt.pause(0.01)