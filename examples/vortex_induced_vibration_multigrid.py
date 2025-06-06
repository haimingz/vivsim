# This example simulates the vortex-induced vibration of a circular cylinder
# in a 2D flow using the Immersed Boundary Method (IBM) coupled with the
# Lattice Boltzmann Method (LBM). The cylinder is placed at the center of the
# domain and is free to move in the flow. The flow is driven by a constant
# velocity U0 in the x direction.

import math
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

from vivsim import dyn, ib, lbm, multigrid as mg, post, mrt, multidevice as md

# ============================= plot options =======================

PLOT = True  # whether to plot the results
PLOT_EVERY = 100  # plot every n time steps
PLOT_AFTER = 00  # plot after n time steps

# ====================== Configuration ======================

# LBM parameters
D = 48                 # Cylinder diameter
U0 = 0.1               # Inlet velocity
TM = 60000             # Total time steps

# multi-block config
HEIGHT = 20 * D
WIDTHS = jnp.array([7.5, 0.5, 4, 0.5, 17.5]) * D
LEVELS = [-2, -1, 0, -1, -2]

# cylinder position
X_OBJ = 10 * D
Y_OBJ = HEIGHT // 2 

# IB parameters
N_MARKER = 4 * D
N_ITER_MDF = 3
IB_MARGIN = 4          # Margin of the IB region to the cylinder

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
MRT_COL_LEFT0, MRT_SRC_LEFT0 = mrt.precompute_left_matrices(mg.get_omega(NU, LEVELS[0]))
MRT_COL_LEFT1, MRT_SRC_LEFT1 = mrt.precompute_left_matrices(mg.get_omega(NU, LEVELS[1]))
MRT_COL_LEFT2, MRT_SRC_LEFT2 = mrt.precompute_left_matrices(mg.get_omega(NU, LEVELS[2]))
MRT_COL_LEFT3, MRT_SRC_LEFT3 = mrt.precompute_left_matrices(mg.get_omega(NU, LEVELS[3]))
MRT_COL_LEFT4, MRT_SRC_LEFT4 = mrt.precompute_left_matrices(mg.get_omega(NU, LEVELS[3]))

# ====================== IBM parameters ==================

X_OBJ_LOCAL = X_OBJ - WIDTHS[0] - WIDTHS[1]
Y_OBJ_LOCAL = Y_OBJ

THETA_MAKERS = jnp.linspace(0, jnp.pi * 2, N_MARKER, dtype=jnp.float32, endpoint=False)
X_MARKERS_LOCAL = X_OBJ_LOCAL + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS_LOCAL = Y_OBJ_LOCAL + 0.5 * D * jnp.sin(THETA_MAKERS)

L_ARC = D * math.pi / N_MARKER  # arc length between the markers

# dynamic ibm region
IB_START_X = int(X_OBJ_LOCAL - 0.5 * D - IB_MARGIN)
IB_START_Y = int(Y_OBJ_LOCAL - 0.5 * D - IB_MARGIN)
IB_SIZE = D + IB_MARGIN * 2
 
X, Y = jnp.meshgrid(jnp.arange(WIDTHS[2]), jnp.arange(HEIGHT), indexing='ij')


# ======================= define variables =====================

f0, rho0, u0 = mg.init_grid(WIDTHS[0], HEIGHT, LEVELS[0], buffer_x=1)
f1, rho1, u1 = mg.init_grid(WIDTHS[1], HEIGHT, LEVELS[1], buffer_x=1)
f2, rho2, u2 = mg.init_grid(WIDTHS[2], HEIGHT, LEVELS[2])
f3, rho3, u3 = mg.init_grid(WIDTHS[3], HEIGHT, LEVELS[3], buffer_x=1)
f4, rho4, u4 = mg.init_grid(WIDTHS[4], HEIGHT, LEVELS[4], buffer_x=1)

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

d = jnp.zeros((2), dtype=jnp.float32) 
v = jnp.zeros((2), dtype=jnp.float32) 
a = jnp.zeros((2), dtype=jnp.float32) 
h = jnp.zeros((2), dtype=jnp.float32) 

v = v.at[1].set(U0 * 0.01) 

feq_init = f0[:, 1, 1]

# ======================= compute routine =====================


def macro_collision(f, left_matrix):  
    rho, u = lbm.get_macroscopic(f[:, 1:-1, 1:-1])
    feq = lbm.get_equilibrium(rho, u)
    f = f.at[:, 1:-1, 1:-1].set(mrt.collision(f[:, 1:-1, 1:-1], feq, left_matrix))
    return f, rho, u

def solve_fsi(f, u, d, v, a, h):
    
    # update markers position
    x_markers, y_markers = ib.get_markers_coords_2dof(X_MARKERS_LOCAL, Y_MARKERS_LOCAL, d)
    
    # update ibm region
    ib_start_x = (IB_START_X + d[0]).astype(jnp.int32)
    ib_start_y = (IB_START_Y + d[1]).astype(jnp.int32)
    
    # extract data from ibm region
    u_slice = jax.lax.dynamic_slice(u, (0, ib_start_x, ib_start_y), (2, IB_SIZE, IB_SIZE))
    X_slice = jax.lax.dynamic_slice(X, (ib_start_x, ib_start_y), (IB_SIZE, IB_SIZE))
    Y_slice = jax.lax.dynamic_slice(Y, (ib_start_x, ib_start_y), (IB_SIZE, IB_SIZE))
    f_slice = jax.lax.dynamic_slice(f, (0, ib_start_x + 1, ib_start_y + 1), (9, IB_SIZE, IB_SIZE))
    
    # calculate ibm force
    g_slice, h_markers = ib.multi_direct_forcing(u_slice, X_slice, Y_slice, 
                                                   v, x_markers, y_markers, N_MARKER, L_ARC, 
                                                   N_ITER_MDF, ib.kernel_range4)
    
    # apply the force to the lattice
    g_lattice = lbm.get_discretized_force(g_slice, u_slice)
    s_slice = mrt.get_source(g_lattice, MRT_SRC_LEFT3)    
    f = jax.lax.dynamic_update_slice(f, f_slice + s_slice, (0, ib_start_x + 1, ib_start_y + 1))

    # apply the force to the cylinder
    h = ib.get_force_to_obj(h_markers)
    h += a * math.pi * D ** 2 / 4   
    a, v, d = dyn.newmark_2dof(a, v, d, h, MASS, STIFFNESS, DAMPING)
    
    return f, d, v, a, h

def update_coarse0(f1, f2, f3, d, v, a, h):
    
    f2, rho2, u2 = macro_collision(f2, MRT_COL_LEFT2)
    f2, d, v, a, h = solve_fsi(f2, u2, d, v, a, h)
    
    f2 = lbm.streaming(f2)
    
    f2 = mg.coarse_to_fine(f1, f2, dir='right')
    f2 = mg.coarse_to_fine(f3, f2, dir='left')
    
    return f2, rho2, u2, d, v, a, h

def update_coarse1(f1, f2, f3, f4, d, v, a, h):
    
    # collision (mesh1 & mesh3)
    f1, rho1, u1 = macro_collision(f1, MRT_COL_LEFT1)
    f3, rho3, u3 = macro_collision(f3, MRT_COL_LEFT3)
    
    # streaming
    f1 = lbm.streaming(f1)
    f3 = lbm.streaming(f3)
    
    # update fine mesh twice (mesh2)
    f2, rho2, u2, d, v, a, h = update_coarse0(f1, f2, f3, d, v, a, h)
    f2, rho2, u2, d, v, a, h = update_coarse0(f1, f2, f3, d, v, a, h)
    
    # boundary conditions
    f1 = mg.fine_to_coarse(f2, f1, dir='left')
    f1 = mg.coarse_to_fine(f0, f1, dir='right')
    
    f3 = mg.fine_to_coarse(f2, f3, dir='right')
    f3 = mg.coarse_to_fine(f4, f3, dir='left')    
    
    return (f1, rho1, u1, 
            f2, rho2, u2,
            f3, rho3, u3, 
            d, v, a, h,)

@jax.jit
def update_coarse2(f0, f1, f2, f3, f4, d, v, a, h):
    
    # collision (mesh0 & mesh4)
    f0, rho0, u0 = macro_collision(f0, MRT_COL_LEFT0)
    f4, rho4, u4 = macro_collision(f4, MRT_COL_LEFT4)
    
    # streaming (mesh0 & mesh4)
    f0 = lbm.streaming(f0)
    f4 = lbm.streaming(f4)
    
    # update fine mesh twice (mesh1 & mesh2 & mesh3)
    (f1, rho1, u1, 
     f2, rho2, u2,
     f3, rho3, u3, 
     d, v, a, h) = update_coarse1(f1, f2, f3, f4, d, v, a, h)
    (f1, rho1, u1, 
     f2, rho2, u2,
     f3, rho3, u3, 
     d, v, a, h) = update_coarse1(f1, f2, f3, f4, d, v, a, h)

    # boundary conditions
    f0 = mg.fine_to_coarse(f1, f0, dir='left')
    f0 = lbm.velocity_boundary(f0, U0, 0, loc='left')
    
    f4 = mg.fine_to_coarse(f3, f4, dir='right') 
    f4 = lbm.boundary_equilibrium(f4, feq_init[:, jnp.newaxis], loc='right')
    
    return (f0, rho0, u0,
            f1, rho1, u1, 
            f2, rho2, u2, 
            f3, rho3, u3, 
            f4, rho4, u4, 
            d, v, a, h)


# ======================= create plot template =====================

if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    
    plt.figure(figsize=(10, 4))

    kwargs = dict(
        cmap="seismic", aspect="equal", origin="lower",
        # norm=mpl.colors.CenteredNorm(),
        vmax=0.01,
        vmin=-0.01,
    )

    im0 = plt.imshow(
        post.calculate_curl(u0).T, 
        extent=[0, WIDTHS[0] / D, 0, HEIGHT / D],
        **kwargs
    )
    
    im1 = plt.imshow(
        post.calculate_curl(u1).T * 2, 
        extent=[WIDTHS[0] / D, jnp.sum(WIDTHS[:2]) / D, 0, HEIGHT / D],
        **kwargs
    )
    
    im2 = plt.imshow(
        post.calculate_curl(u2).T, 
        extent=[jnp.sum(WIDTHS[:2]) / D, jnp.sum(WIDTHS[:3]) / D, 0, HEIGHT / D],
        **kwargs
    )
    
    im3 = plt.imshow(
        post.calculate_curl(u3).T,
        extent=[ jnp.sum(WIDTHS[:3]) / D,  jnp.sum(WIDTHS[:4]) / D, 0, HEIGHT / D],
        **kwargs
    )

    im4 = plt.imshow(
        post.calculate_curl(u4).T,
        extent=[jnp.sum(WIDTHS[:4]) / D, jnp.sum(WIDTHS)/D, 0, HEIGHT / D],
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
    plt.axvline(jnp.sum(WIDTHS[:1])/ D, color="g", linestyle="--", linewidth=0.5)
    plt.axvline(jnp.sum(WIDTHS[:2]) / D, color="g", linestyle="--", linewidth=0.5)   
    plt.axvline(jnp.sum(WIDTHS[:3]) / D, color="g", linestyle="--", linewidth=0.5)
    plt.axvline(jnp.sum(WIDTHS[:4]) / D, color="g", linestyle="--", linewidth=0.5)

    # mark the initial position of the cylinder
    plt.plot(X_OBJ / D, Y_OBJ / D, 
             marker='+', markersize=10, color='k', linestyle='None', markeredgewidth=0.5)
    
    plt.tight_layout()
   

# ========================== start simulation ==========================

for t in tqdm(range(TM)):
    (
        f0, rho0, u0, 
        f1, rho1, u1, 
        f2, rho2, u2, 
        f3, rho3, u3, 
        f4, rho4, u4, 
        d, v, a, h
    ) = update_coarse2(
        f0, f1, f2, f3, f4, d, v, a, h,
    )
    
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:

        im0.set_data(post.calculate_curl(u0).T / D / U0 / 4)
        im1.set_data(post.calculate_curl(u1).T / D / U0 / 2)
        im2.set_data(post.calculate_curl(u2).T / D / U0 / 1)
        im3.set_data(post.calculate_curl(u3).T / D / U0 / 2)   
        im4.set_data(post.calculate_curl(u4).T / D / U0 / 4)  
        circle.center = ((X_OBJ + d[0]) / D, (Y_OBJ + d[1]) / D)
        plt.pause(0.01)