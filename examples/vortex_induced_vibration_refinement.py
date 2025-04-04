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

# ========================== CONSTANTS =======================

D = 20                 # Cylinder diameter
U0 = 0.1                # Inlet velocity

# Cylinder position
X_OBJ = 10 * D          # Cylinder x position
Y_OBJ = 10 * D          # Cylinder y position

# IB method parameters
N_MARKER = 4 * D       # Number of markers on cylinder
N_ITER_MDF = 3         # Multi-direct forcing iterations
IB_PADDING = 2         # Padding around the cylinder

# Physical parameters
RE = 200               # Reynolds number
UR = 5                 # Reduced velocity
MR = 10                # Mass ratio
DR = 0                 # Damping ratio

# ========================== PLOT OPTIONS =======================

PLOT = True
PLOT_AFTER = 0
PLOT_EVERY = 50

# ========================== GRID PARAMETERS =======================

# absolute coordinates of the grids (G1, G2, G3)

G1_LEVEL = -2                           # level of refinement (coarsest)
G1_HEIGHT = 20 * D
G1_WIDTH = 30 * D

G2_LEVEL = -1                           # level of refinement (mild)
G2_HEIGHT = 8 * D 
G2_WIDTH = 12 * D
G2_X1 = 6 * D                           # left boundary of grid 2
G2_X2 = G2_X1 + G2_WIDTH                # right boundary of grid 2
G2_Y1 = (G1_HEIGHT - G2_HEIGHT) // 2    # top boundary of grid 2
G2_Y2 = G2_Y1 + G2_HEIGHT               # bottom boundary of grid 2

G3_LEVEL = 0                            # level of refinement (finest)
G3_HEIGHT = 6 * D 
G3_WIDTH = 8 * D
G3_X1 = 8 * D                           # left boundary of grid 3
G3_X2 = G3_X1 + G3_WIDTH                # right boundary of grid 3
G3_Y1 = (G1_HEIGHT - G3_HEIGHT) // 2    # top boundary of grid 3
G3_Y2 = G3_Y1 + G3_HEIGHT               # bottom boundary of grid 3

# relative coordinates of the grids (with respect to the parent grid)

def global_to_grid1(x, y):
    return int(x // 4), int(y // 4) 

def global_to_grid2(x, y):
    return int((x - G2_X1) // 2), int((y - G2_Y1) // 2)

def global_to_grid3(x, y):
    return int(x - G3_X1), int(y - G3_Y1)

G2_X1_, G2_Y1_ = global_to_grid1(G2_X1, G2_Y1)
G2_X2_, G2_Y2_ = global_to_grid1(G2_X2, G2_Y2)
G3_X1_, G3_Y1_ = global_to_grid2(G3_X1, G3_Y1)
G3_X2_, G3_Y2_ = global_to_grid2(G3_X2, G3_Y2)

# ==================== DERIVED CONSTANTS =======================

# structural parameters
FN = U0 / (UR * D)                                          # Natural frequency
MASS = math.pi * (D / 2) ** 2 * MR                          # Mass of the cylinder
STIFFNESS = (FN * 2 * math.pi) ** 2 * MASS * (1 + 1 / MR)   # Stiffness of the spring
DAMPING = 2 * math.sqrt(STIFFNESS * MASS) * DR              # Damping of the spring

# time steps
TM = int(50 / FN / 4)                                       # Total time steps 

# fluid parameters
NU = U0 * D / RE                                            # Kinematic viscosity
MRT_COL_LEFT1, MRT_SRC_LEFT1 = mrt.precompute_left_matrices(mg.get_omega(NU, G1_LEVEL))
MRT_COL_LEFT2, MRT_SRC_LEFT2 = mrt.precompute_left_matrices(mg.get_omega(NU, G2_LEVEL))
MRT_COL_LEFT3, MRT_SRC_LEFT3 = mrt.precompute_left_matrices(mg.get_omega(NU, G3_LEVEL))

# IBM parameters
X_OBJ_, Y_OBJ_ = global_to_grid3(X_OBJ , Y_OBJ)
THETA_MAKERS = jnp.linspace(0, jnp.pi * 2, N_MARKER, dtype=jnp.float32, endpoint=False)
X_MARKERS_ = X_OBJ_ + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS_ = Y_OBJ_ + 0.5 * D * jnp.sin(THETA_MAKERS)
L_ARC = D * math.pi / N_MARKER  # arc length between the markers

# dynamic ibm region
IB_X1_ = int(X_OBJ_ - 0.5 * D - IB_PADDING)
IB_Y1_ = int(Y_OBJ_ - 0.5 * D - IB_PADDING)
IB_SIZE = D + IB_PADDING * 2

# coords of grid3 (where FSI happens)
G3_X_, G3_Y_ = jnp.mgrid[0:G3_WIDTH, 0:G3_HEIGHT]

# ======================= define variables =====================

f1, rho1, u1 = mg.init_grid(G1_WIDTH, G1_HEIGHT, G1_LEVEL)
f2, rho2, u2 = mg.init_grid(G2_WIDTH, G2_HEIGHT, G2_LEVEL)
f3, rho3, u3 = mg.init_grid(G3_WIDTH, G3_HEIGHT, G3_LEVEL)

# ======================= initial conditions =====================

u1 = u1.at[0].set(U0)
u2 = u2.at[0].set(U0)
u3 = u3.at[0].set(U0)

f1 =  lbm.get_equilibrium(rho1, u1)
f2 =  lbm.get_equilibrium(rho2, u2)
f3 =  lbm.get_equilibrium(rho3, u3)

d = jnp.zeros((2), dtype=jnp.float32) 
v = jnp.zeros((2), dtype=jnp.float32) 
a = jnp.zeros((2), dtype=jnp.float32) 
h = jnp.zeros((2), dtype=jnp.float32) 

v = v.at[1].set(U0 * 0.01) 

feq_init = f1[:, 1, 1]

# ======================= compute routine =====================

def macro_collision(f, left_matrix):  
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = mrt.collision(f, feq, left_matrix)
    return f, rho, u


def solve_fsi(f, u, d, v, a, h, left_matrix):
    
    # update markers position
    x_markers, y_markers = ib.get_markers_coords_2dof(X_MARKERS_, Y_MARKERS_, d)
    
    # update ibm region
    ib_x1_ = (IB_X1_ + d[0]).astype(jnp.int32)
    ib_y1_ = (IB_Y1_ + d[1]).astype(jnp.int32)
    
    # extract data from ibm region
    u_slice = jax.lax.dynamic_slice(u, (0, ib_x1_, ib_y1_), (2, IB_SIZE, IB_SIZE))
    X_slice = jax.lax.dynamic_slice(G3_X_, (ib_x1_, ib_y1_), (IB_SIZE, IB_SIZE))
    Y_slice = jax.lax.dynamic_slice(G3_Y_, (ib_x1_, ib_y1_), (IB_SIZE, IB_SIZE))
    f_slice = jax.lax.dynamic_slice(f, (0, ib_x1_, ib_y1_), (9, IB_SIZE, IB_SIZE))
    
    # calculate ibm force
    g_slice, h_markers = ib.multi_direct_forcing(u_slice, X_slice, Y_slice, 
                                                   v, x_markers, y_markers, N_MARKER, L_ARC, 
                                                   N_ITER_MDF, ib.kernel_range3)

    # apply the force to the lattice
    g_lattice = lbm.get_discretized_force(g_slice, u_slice)
    s_slice = mrt.get_source(g_lattice, left_matrix)    
    f = jax.lax.dynamic_update_slice(f, f_slice + s_slice, (0, ib_x1_, ib_y1_))

    # apply the force to the cylinder
    h = ib.get_force_to_obj(h_markers)
    h += a * math.pi * D ** 2 / 4   
    a, v, d = dyn.newmark_2dof(a, v, d, h, MASS, STIFFNESS, DAMPING)
    
    return f, d, v, a, h


def update_grid3(f2, f3, d, v, a, h):
    
    # f3, rho3, u3 = macro_collision(f3, MRT_COL_LEFT3)
    rho3, u3 = lbm.get_macroscopic(f3)
    feq3 = lbm.get_equilibrium(rho3, u3)
    f3 = mrt.collision(f3, feq3, MRT_COL_LEFT3)
    f3, d, v, a, h = solve_fsi(f3, u3, d, v, a, h, MRT_SRC_LEFT3)    
    f3 = lbm.streaming(f3)
    
    # outer boundary
    f3 = mg.coarse_to_fine(f2[:, G3_X1_, None, G3_Y1_: G3_Y2_], f3, dir='right')
    f3 = mg.coarse_to_fine(f2[:, G3_X2_ - 1, None, G3_Y1_: G3_Y2_], f3, dir='left')
    f3 = mg.coarse_to_fine(f2[:, G3_X1_: G3_X2_, G3_Y2_ - 1, None], f3, dir='down')
    f3 = mg.coarse_to_fine(f2[:, G3_X1_: G3_X2_, G3_Y1_, None], f3, dir='up')
    
    return f3, rho3, u3, d, v, a, h


def update_grid2(f1, f2, f3, d, v, a, h):
    
    f2, rho2, u2 = macro_collision(f2, MRT_COL_LEFT2)
    f2 = lbm.streaming(f2)
    
    # outer boundary
    f2 = mg.coarse_to_fine(f1[:, G2_X1_, None, G2_Y1_: G2_Y2_], f2, dir='right')
    f2 = mg.coarse_to_fine(f1[:, G2_X2_ - 1, None, G2_Y1_: G2_Y2_], f2, dir='left')
    f2 = mg.coarse_to_fine(f1[:, G2_X1_: G2_X2_, G2_Y2_ - 1, None], f2, dir='down')
    f2 = mg.coarse_to_fine(f1[:, G2_X1_: G2_X2_, G2_Y1_, None], f2, dir='up')
    
    # inner boundary
    f3, rho3, u3, d, v, a, h = update_grid3(f2, f3, d, v, a, h)
    f3, rho3, u3, d, v, a, h = update_grid3(f2, f3, d, v, a, h)
    
    f2 = f2.at[:, G3_X1_, None, G3_Y1_: G3_Y2_].set(mg.fine_to_coarse(f3, f2[:, G3_X1_, None, G3_Y1_: G3_Y2_], dir='left'))
    f2 = f2.at[:, G3_X2_ - 1, None, G3_Y1_: G3_Y2_].set(mg.fine_to_coarse(f3, f2[:, G3_X2_ - 1, None, G3_Y1_: G3_Y2_], dir='right'))
    f2 = f2.at[:, G3_X1_: G3_X2_, G3_Y2_ - 1, None].set(mg.fine_to_coarse(f3, f2[:, G3_X1_: G3_X2_, G3_Y2_ - 1, None], dir='up'))
    f2 = f2.at[:, G3_X1_: G3_X2_, G3_Y1_, None].set(mg.fine_to_coarse(f3, f2[:, G3_X1_: G3_X2_, G3_Y1_, None], dir='down'))

    return f2, rho2, u2, f3, rho3, u3, d, v, a, h


@jax.jit
def update_grid1(f1, f2, f3, d, v, a, h):
    
    f1, rho1, u1 = macro_collision(f1, MRT_COL_LEFT1)
    f1 = lbm.streaming(f1)
    
    # outer boundary
    f1 = lbm.velocity_boundary(f1, U0, 0, loc='left')
    f1 = lbm.boundary_equilibrium(f1, feq_init[:, jnp.newaxis], loc='right')
    
    # inner boundary
    f2, rho2, u2, f3, rho3, u3, d, v, a, h = update_grid2(f1, f2, f3, d, v, a, h)
    f2, rho2, u2, f3, rho3, u3, d, v, a, h = update_grid2(f1, f2, f3, d, v, a, h)
    
    f1 = f1.at[:, G2_X1_, None, G2_Y1_: G2_Y2_].set(mg.fine_to_coarse(f2, f1[:, G2_X1_, None, G2_Y1_: G2_Y2_], dir='left'))
    f1 = f1.at[:, G2_X2_ - 1, None, G2_Y1_: G2_Y2_].set(mg.fine_to_coarse(f2, f1[:, G2_X2_ - 1, None, G2_Y1_: G2_Y2_], dir='right'))
    f1 = f1.at[:, G2_X1_: G2_X2_, G2_Y2_ - 1, None].set(mg.fine_to_coarse(f2, f1[:, G2_X1_: G2_X2_, G2_Y2_ - 1, None], dir='up'))
    f1 = f1.at[:, G2_X1_: G2_X2_, G2_Y1_, None].set(mg.fine_to_coarse(f2, f1[:, G2_X1_: G2_X2_, G2_Y1_, None], dir='down'))
    
    return (f1, rho1, u1, 
            f2, rho2, u2,
            f3, rho3, u3,
            d, v, a, h,)


# ======================= create plot template =====================

if PLOT:
    
    mpl.rcParams['figure.raise_window'] = False
    
    plt.figure(figsize=(10, 4))
    
    # -------------------- grid lines --------------------
    # Grid 1 grid lines
    for x in range(0, G1_WIDTH // 4 + 1):
        plt.plot([x * 4 / D, x * 4 / D], [0, G1_HEIGHT / D], 'k', linewidth=0.05)
    for y in range(0, G1_HEIGHT // 4 + 1):
        plt.plot([0, G1_WIDTH / D], [y * 4 / D, y * 4 / D], 'k', linewidth=0.05)
    # add a label at bottom-left corner of grid 1
    plt.text(0.1, 0.1, 'Grid 1',
             horizontalalignment='left', verticalalignment='bottom',
             fontsize=6, color='black')

    # Grid 2 grid lines
    for x in range(G2_X1 // 2 , G2_X2 // 2 + 1):
        plt.plot([x * 2 / D, x * 2 / D], [G2_Y1 / D, G2_Y2 / D], 'k', linewidth=0.05)
    for y in range(G2_Y1 // 2, G2_Y2 // 2 + 1):
        plt.plot([G2_X1 / D, G2_X2 / D], [y * 2 / D, y * 2 / D], 'k', linewidth=0.05)
    # add a label at bottom-left corner of grid 2
    plt.text(G2_X1 / D + 0.1, G2_Y1 / D + 0.1, 'Grid 2',
             horizontalalignment='left', verticalalignment='bottom',
             fontsize=6, color='black')
    
    # Grid 3 grid lines
    for x in range(G3_X1, G3_X2 + 1):
        plt.plot([x / D, x / D], [G3_Y1 / D, G3_Y2 / D], 'k', linewidth=0.05)
    for y in range(G3_Y1, G3_Y2 + 1):
        plt.plot([G3_X1 / D, G3_X2 / D], [y / D, y / D], 'k', linewidth=0.05)
    # add a label at bottom-left corner of grid 3
    plt.text(G3_X1 / D + 0.1, G3_Y1 / D + 0.1, 'Grid 3',
                horizontalalignment='left', verticalalignment='bottom',
                fontsize=6, color='black')
    
    # -------------------- immersed boundary region --------------------
    ib_start_x = (G3_X1 + IB_X1_ + d[0]).astype(jnp.int32)
    ib_start_y = (G3_Y1 + IB_Y1_ + d[1]).astype(jnp.int32)
    ib_region = plt.Rectangle((ib_start_x / D, ib_start_y / D),
                               IB_SIZE / D, IB_SIZE / D,
                               edgecolor='blue', linewidth=0.5, fill=False)
    plt.gca().add_patch(ib_region)
    ib_label = plt.text(ib_start_x / D + 0.5, ib_start_y / D + 2, 'IB Region',
                        horizontalalignment='center', verticalalignment='bottom',
                        fontsize=6, color='blue')
    
    
    # -------------------- plot the curl of the velocity field --------------------
    kwargs = dict(cmap="seismic", aspect="equal", origin="lower",
        vmax=0.2, vmin=-0.2)
    
    im1 = plt.imshow(post.calculate_curl(u1).T * 0,
                     extent=[0, G1_WIDTH / D, 0, G1_HEIGHT / D], **kwargs)
    im2 = plt.imshow(post.calculate_curl(u2).T,
                     extent=[G2_X1 / D, G2_X2 / D,
                             G2_Y1 / D, G2_Y2 / D], **kwargs)
    im3 = plt.imshow(post.calculate_curl(u3).T, 
                     extent=[G3_X1 / D, G3_X2 / D, 
                             G3_Y1 / D, G3_Y2 / D], **kwargs)
    plt.colorbar()
    
   
    # draw a circle representing the cylinder
    cylinder = plt.Circle(((X_OBJ + d[0]) / D , (Y_OBJ + d[1]) / D), 0.5, 
                        edgecolor='black', linewidth=0.5, facecolor='white', fill=True)
    plt.gca().add_artist(cylinder)
    
    # mark the initial position of the cylinder
    plt.plot(X_OBJ / D , Y_OBJ / D, marker='+', markersize=10, color='k', linestyle='None', markeredgewidth=0.5)
       
        
    plt.xlabel("x/D")
    plt.ylabel("y/D")
    plt.tight_layout()


# ========================== start simulation ==========================

for t in tqdm(range(TM)):
    (
        f1, rho1, u1, 
        f2, rho2, u2,
        f3, rho3, u3,
        d, v, a, h,
    ) = update_grid1(
        f1, f2, f3, d, v, a, h,
    )
    
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:

        im1.set_data(post.calculate_curl(u1).T)
        im2.set_data(post.calculate_curl(u2).T * 2)
        im3.set_data(post.calculate_curl(u3).T * 4)            
    
        obj_x = (X_OBJ + d[0]) / D
        obj_y = (Y_OBJ + d[1]) / D
        cylinder.set_center((obj_x, obj_y))
        
        ib_region.set_xy(((G3_X1 + IB_X1_ + d[0]).astype(jnp.int32) / D, (G3_Y1 + IB_Y1_ + d[1]).astype(jnp.int32) / D))
        ib_label.set_position((obj_x, obj_y + 0.6))
        plt.pause(0.01)
