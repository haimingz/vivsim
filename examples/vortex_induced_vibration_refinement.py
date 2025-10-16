"""
Vortex-Induced Vibration (VIV) with Multigrid Refinement

This example simulates the vortex-induced vibration of a circular cylinder
in a 2D flow using the Immersed Boundary Method (IBM) coupled with the
Lattice Boltzmann Method (LBM) with adaptive mesh refinement. The cylinder
is free to oscillate in both cross-flow and in-line directions with 
spring-mass-damper constraints. The simulation uses three nested grids 
with different levels of refinement for computational efficiency.
"""

import math

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

from vivsim import dyn, ib, lbm, post
from vivsim import multigrid as mg



# ========================== VISUALIZATION PARAMETERS ====================

PLOT = True                 # Enable real-time visualization
PLOT_EVERY = 50             # Plot update frequency
PLOT_AFTER = 0              # Start plotting after this timestep


# ========================== PHYSICAL PARAMETERS =====================

RE = 200                    # Reynolds number
UR = 5                      # Reduced velocity
MR = 10                     # Mass ratio
DR = 0                      # Damping ratio


# ========================== GEOMETRY PARAMETERS =======================

D = 40                      # Cylinder diameter
OBJ_X = 10 * D              # Cylinder center x-position (absolute)
OBJ_Y = 10 * D              # Cylinder center y-position (absolute)
N_MARKER = 4 * D            # Number of markers on cylinder surface

# Grid 1 (coarsest level)
G1_LEVEL = -2               # Refinement level
G1_WIDTH = 30 * D           # Grid width
G1_HEIGHT = 20 * D          # Grid height

# Grid 2 (intermediate level) - absolute coordinates
G2_LEVEL = -1               # Refinement level
G2_WIDTH = 12 * D           # Grid width
G2_HEIGHT = 8 * D           # Grid height
G2_X1 = 6 * D               # Left boundary
G2_X2 = G2_X1 + G2_WIDTH    # Right boundary
G2_Y1 = (G1_HEIGHT - G2_HEIGHT) // 2  # Bottom boundary
G2_Y2 = G2_Y1 + G2_HEIGHT   # Top boundary

# Grid 3 (finest level) - absolute coordinates
G3_LEVEL = 0                # Refinement level
G3_WIDTH = 8 * D            # Grid width
G3_HEIGHT = 6 * D           # Grid height
G3_X1 = 8 * D               # Left boundary
G3_X2 = G3_X1 + G3_WIDTH    # Right boundary
G3_Y1 = (G1_HEIGHT - G3_HEIGHT) // 2  # Bottom boundary
G3_Y2 = G3_Y1 + G3_HEIGHT   # Top boundary

# Relative coordinates (with respect to parent grid)
G2_X1_, G2_Y1_ = mg.coord_to_indices(G2_X1, G2_Y1, 0, 0, -2)
G2_X2_, G2_Y2_ = mg.coord_to_indices(G2_X2, G2_Y2, 0, 0, -2)
G3_X1_, G3_Y1_ = mg.coord_to_indices(G3_X1, G3_Y1, G2_X1, G2_Y1, -1)
G3_X2_, G3_Y2_ = mg.coord_to_indices(G3_X2, G3_Y2, G2_X1, G2_Y1, -1)
OBJ_X_, OBJ_Y_ = mg.coord_to_indices(OBJ_X, OBJ_Y, G3_X1, G3_Y1, 0)


# ========================== SIMULATION PARAMETERS =======================

U0 = 0.05                                           # Inlet velocity
NU = U0 * D / RE                                    # Kinematic viscosity
FN = U0 / (UR * D)                                  # Natural frequency
AREA = math.pi * (D / 2) ** 2                       # Cylinder cross-sectional area
M = AREA * MR                                       # Mass of the cylinder
K = (FN * 2 * math.pi) ** 2 * M                     # Spring stiffness
C = 2 * math.sqrt(K * M) * DR                       # Damping coefficient
TM = int(100 / FN / 4)                              # Total simulation timesteps


# ========================== IB-LBM PARAMETERS ==========================

# Relaxation parameters for each grid level
OMEGA1 = mg.get_omega(NU, -2)           # Grid 1 (coarsest) relaxation parameter
OMEGA2 = mg.get_omega(NU, -1)           # Grid 2 (intermediate) relaxation parameter
OMEGA3 = mg.get_omega(NU, 0)            # Grid 3 (finest) relaxation parameter

# IBM marker configuration
THETA = jnp.linspace(0, 2 * jnp.pi, N_MARKER, dtype=jnp.float32, endpoint=False)
X_MARKERS = OBJ_X_ + 0.5 * D * jnp.cos(THETA)       # Initial x positions (relative to Grid 3)
Y_MARKERS = OBJ_Y_ + 0.5 * D * jnp.sin(THETA)       # Initial y positions (relative to Grid 3)
DS_MARKERS = D * math.pi / N_MARKER                    # Arc length between markers

# IBM region configuration
IB_PAD = 10                                          # Padding around cylinder for IBM region
IB_X1_ = int(OBJ_X_ - 0.5 * D - IB_PAD)            # Initial x position of IB region
IB_Y1_ = int(OBJ_Y_ - 0.5 * D - IB_PAD)            # Initial y position of IB region
IB_SIZE = D + IB_PAD * 2                           # Size of IB region

IB_ITER = 3                                        # Multi-direct forcing iterations
FSI_ITER = 3                                       # FSI sub-iterations per time step

# Grid 3 coordinates for IB calculations
G3_X_, G3_Y_ = jnp.mgrid[0:G3_WIDTH, 0:G3_HEIGHT]


# ======================= INITIALIZE VARIABLES ====================

# Initialize grids
f1, rho1, u1 = mg.init_grid(G1_WIDTH, G1_HEIGHT, G1_LEVEL)
f2, rho2, u2 = mg.init_grid(G2_WIDTH, G2_HEIGHT, G2_LEVEL)
f3, rho3, u3 = mg.init_grid(G3_WIDTH, G3_HEIGHT, G3_LEVEL)

# Set initial velocity
u1 = u1.at[0].set(U0)
u2 = u2.at[0].set(U0)
u3 = u3.at[0].set(U0)

# Initialize distribution functions
f1 = lbm.get_equilibrium(rho1, u1)
f2 = lbm.get_equilibrium(rho2, u2)
f3 = lbm.get_equilibrium(rho3, u3)

# Structural variables
d = jnp.zeros(2, dtype=jnp.float32)     # Displacement of cylinder
v = jnp.zeros(2, dtype=jnp.float32)     # Velocity of cylinder
a = jnp.zeros(2, dtype=jnp.float32)     # Acceleration of cylinder
h = jnp.zeros(2, dtype=jnp.float32)     # Hydrodynamic force

# Optional: add initial perturbation
v = v.at[1].set(0.01 * U0)


# ======================= SIMULATION ROUTINES =====================

def solve_fsi(f, rho, u, d, v, a, h):
    """Solve fluid-structure interaction using IBM with FSI iterations."""
    
    # Update IBM region boundaries (dynamic region follows cylinder)
    ib_x1_ = (IB_X1_ + d[0]).astype(jnp.int32)
    ib_y1_ = (IB_Y1_ + d[1]).astype(jnp.int32)
    
    # Extract data from IBM region for efficient computation
    rho_ib = jax.lax.dynamic_slice(rho, (ib_x1_, ib_y1_), (IB_SIZE, IB_SIZE))
    u_ib = jax.lax.dynamic_slice(u, (0, ib_x1_, ib_y1_), (2, IB_SIZE, IB_SIZE))
    x_ib = jax.lax.dynamic_slice(G3_X_, (ib_x1_, ib_y1_), (IB_SIZE, IB_SIZE))
    y_ib = jax.lax.dynamic_slice(G3_Y_, (ib_x1_, ib_y1_), (IB_SIZE, IB_SIZE))
    f_ib = jax.lax.dynamic_slice(f, (0, ib_x1_, ib_y1_), (9, IB_SIZE, IB_SIZE))
    a_old, v_old, d_old = a, v, d
    
    for i in range(FSI_ITER):
        
        # Update marker positions based on cylinder displacement
        x_markers, y_markers = dyn.get_markers_coords_2dof(X_MARKERS, Y_MARKERS, d)
        kernels = ib.get_kernels(x_markers, y_markers, x_ib, y_ib, ib.kernel_range4)
        g_ib, h_marker = ib.multi_direct_forcing(u_ib, v, kernels, DS_MARKERS, n_iter=IB_ITER)
        
        # Apply marker forces to the cylinder with internal mass correction
        h = dyn.get_force_to_obj(h_marker)
        h += a * math.pi * D ** 2 / 4   
        a, v, d = dyn.newmark_2dof(a_old, v_old, d_old, h, M, K, C)
    
    # Update distribution functions and velocity in IBM region using EDM forcing
    f_ib = lbm.forcing_edm(f_ib, g_ib, u_ib, rho_ib)
    f = jax.lax.dynamic_update_slice(f, f_ib, (0, ib_x1_, ib_y1_))
    
    u_ib = u_ib + lbm.get_velocity_correction(g_ib, rho_ib)
    u = jax.lax.dynamic_update_slice(u, u_ib, (0, ib_x1_, ib_y1_))
    
    return f, u, d, v, a, h


def update_grid3(f2, f3, d, v, a, h):
    """Update finest grid (Grid 3) with FSI coupling."""
    
    # Macroscopic variables and collision
    rho3, u3 = lbm.get_macroscopic(f3)
    feq3 = lbm.get_equilibrium(rho3, u3)
    f3 = lbm.collision_kbc(f3, feq3, OMEGA3)
    
    # Solve FSI
    f3, u3, d, v, a, h = solve_fsi(f3, rho3, u3, d, v, a, h)    
    
    # Streaming step
    f3 = lbm.streaming(f3)
    
    # Outer boundary - interpolation from Grid 2
    f3 = mg.coarse_to_fine(f2[:, G3_X1_, None, G3_Y1_: G3_Y2_], f3, dir='right')
    f3 = mg.coarse_to_fine(f2[:, G3_X2_ - 1, None, G3_Y1_: G3_Y2_], f3, dir='left')
    f3 = mg.coarse_to_fine(f2[:, G3_X1_: G3_X2_, G3_Y2_ - 1, None], f3, dir='down')
    f3 = mg.coarse_to_fine(f2[:, G3_X1_: G3_X2_, G3_Y1_, None], f3, dir='up')
    
    return f3, rho3, u3, d, v, a, h


def update_grid2(f1, f2, f3, d, v, a, h):
    """Update intermediate grid (Grid 2)."""
    
    # Macroscopic variables and collision
    rho2, u2 = lbm.get_macroscopic(f2)
    feq2 = lbm.get_equilibrium(rho2, u2)
    f2 = lbm.collision_kbc(f2, feq2, OMEGA2)
    
    # Streaming step
    f2 = lbm.streaming(f2)
    
    # Outer boundary - interpolation from Grid 1
    f2 = mg.coarse_to_fine(f1[:, G2_X1_, None, G2_Y1_: G2_Y2_], f2, dir='right')
    f2 = mg.coarse_to_fine(f1[:, G2_X2_ - 1, None, G2_Y1_: G2_Y2_], f2, dir='left')
    f2 = mg.coarse_to_fine(f1[:, G2_X1_: G2_X2_, G2_Y2_ - 1, None], f2, dir='down')
    f2 = mg.coarse_to_fine(f1[:, G2_X1_: G2_X2_, G2_Y1_, None], f2, dir='up')
    
    # Inner boundary - update Grid 3 twice per Grid 2 update
    f3, rho3, u3, d, v, a, h = update_grid3(f2, f3, d, v, a, h)
    f3, rho3, u3, d, v, a, h = update_grid3(f2, f3, d, v, a, h)
    
    # Update Grid 2 boundaries with Grid 3 data
    f2 = f2.at[:, G3_X1_, None, G3_Y1_: G3_Y2_].set(mg.fine_to_coarse(f3, f2[:, G3_X1_, None, G3_Y1_: G3_Y2_], dir='left'))
    f2 = f2.at[:, G3_X2_ - 1, None, G3_Y1_: G3_Y2_].set(mg.fine_to_coarse(f3, f2[:, G3_X2_ - 1, None, G3_Y1_: G3_Y2_], dir='right'))
    f2 = f2.at[:, G3_X1_: G3_X2_, G3_Y2_ - 1, None].set(mg.fine_to_coarse(f3, f2[:, G3_X1_: G3_X2_, G3_Y2_ - 1, None], dir='up'))
    f2 = f2.at[:, G3_X1_: G3_X2_, G3_Y1_, None].set(mg.fine_to_coarse(f3, f2[:, G3_X1_: G3_X2_, G3_Y1_, None], dir='down'))

    return f2, rho2, u2, f3, rho3, u3, d, v, a, h


@jax.jit
def update_grid1(f1, f2, f3, d, v, a, h):
    """Update coarsest grid (Grid 1) and manage nested grid updates."""
    
    # Macroscopic variables and collision
    rho1, u1 = lbm.get_macroscopic(f1)
    feq1 = lbm.get_equilibrium(rho1, u1)
    f1 = lbm.collision_kbc(f1, feq1, OMEGA1)
    
    # Streaming step
    f1 = lbm.streaming(f1)
    
    # Outer boundary conditions
    f1 = lbm.boundary_nee(f1, loc='left', ux_wall=U0)
    f1 = lbm.boundary_equilibrium(f1, loc='right', ux_wall=U0)
    
    # Inner boundary - update Grid 2 twice per Grid 1 update
    f2, rho2, u2, f3, rho3, u3, d, v, a, h = update_grid2(f1, f2, f3, d, v, a, h)
    f2, rho2, u2, f3, rho3, u3, d, v, a, h = update_grid2(f1, f2, f3, d, v, a, h)
    
    # Update Grid 1 boundaries with Grid 2 data
    f1 = f1.at[:, G2_X1_, None, G2_Y1_: G2_Y2_].set(mg.fine_to_coarse(f2, f1[:, G2_X1_, None, G2_Y1_: G2_Y2_], dir='left'))
    f1 = f1.at[:, G2_X2_ - 1, None, G2_Y1_: G2_Y2_].set(mg.fine_to_coarse(f2, f1[:, G2_X2_ - 1, None, G2_Y1_: G2_Y2_], dir='right'))
    f1 = f1.at[:, G2_X1_: G2_X2_, G2_Y2_ - 1, None].set(mg.fine_to_coarse(f2, f1[:, G2_X1_: G2_X2_, G2_Y2_ - 1, None], dir='up'))
    f1 = f1.at[:, G2_X1_: G2_X2_, G2_Y1_, None].set(mg.fine_to_coarse(f2, f1[:, G2_X1_: G2_X2_, G2_Y1_, None], dir='down'))
    
    return (f1, rho1, u1, 
            f2, rho2, u2,
            f3, rho3, u3,
            d, v, a, h)


# ======================= VISUALIZATION SETUP ====================

mpl.rcParams['figure.raise_window'] = False

# Storage for displacement and force history
d_hist = np.zeros((2, TM))
h_hist = np.zeros((2, TM))

# Setup real-time visualization
if PLOT:
    
    plt.figure(figsize=(10, 4))
    
    # -------------------- Grid lines --------------------
    
    # Grid 1 grid lines (coarsest)
    for x in range(0, G1_WIDTH // 4 + 1):
        plt.plot([x * 4 / D, x * 4 / D], [0, G1_HEIGHT / D], 'k', linewidth=0.05)
    for y in range(0, G1_HEIGHT // 4 + 1):
        plt.plot([0, G1_WIDTH / D], [y * 4 / D, y * 4 / D], 'k', linewidth=0.05)
    plt.text(0.1, 0.1, 'Grid 1',
             horizontalalignment='left', verticalalignment='bottom',
             fontsize=6, color='black')

    # Grid 2 grid lines (intermediate)
    for x in range(G2_X1 // 2, G2_X2 // 2 + 1):
        plt.plot([x * 2 / D, x * 2 / D], [G2_Y1 / D, G2_Y2 / D], 'k', linewidth=0.05)
    for y in range(G2_Y1 // 2, G2_Y2 // 2 + 1):
        plt.plot([G2_X1 / D, G2_X2 / D], [y * 2 / D, y * 2 / D], 'k', linewidth=0.05)
    plt.text(G2_X1 / D + 0.1, G2_Y1 / D + 0.1, 'Grid 2',
             horizontalalignment='left', verticalalignment='bottom',
             fontsize=6, color='black')
    
    # Grid 3 grid lines (finest)
    for x in range(G3_X1, G3_X2 + 1):
        plt.plot([x / D, x / D], [G3_Y1 / D, G3_Y2 / D], 'k', linewidth=0.05)
    for y in range(G3_Y1, G3_Y2 + 1):
        plt.plot([G3_X1 / D, G3_X2 / D], [y / D, y / D], 'k', linewidth=0.05)
    plt.text(G3_X1 / D + 0.1, G3_Y1 / D + 0.1, 'Grid 3',
             horizontalalignment='left', verticalalignment='bottom',
             fontsize=6, color='black')
    
    # -------------------- Immersed boundary region --------------------
    
    ib_start_x = (G3_X1 + IB_X1_ + d[0]).astype(jnp.int32)
    ib_start_y = (G3_Y1 + IB_Y1_ + d[1]).astype(jnp.int32)
    ib_region = plt.Rectangle((ib_start_x / D, ib_start_y / D),
                               IB_SIZE / D, IB_SIZE / D,
                               edgecolor='blue', linewidth=0.5, fill=False)
    plt.gca().add_patch(ib_region)
    ib_label = plt.text(ib_start_x / D + 0.5, ib_start_y / D + 2, 'IB Region',
                        horizontalalignment='center', verticalalignment='bottom',
                        fontsize=6, color='blue')
    
    # -------------------- Vorticity field --------------------
    
    kwargs = dict(cmap="seismic", aspect="equal", origin="lower",
                  vmax=10, vmin=-10)
    
    # Initialize vorticity plots for each grid
    im1 = plt.imshow(post.calculate_vorticity_dimensionless(u1, D / 4, U0).T * 0,
                     extent=[0, G1_WIDTH / D, 0, G1_HEIGHT / D], **kwargs)
    im2 = plt.imshow(post.calculate_vorticity_dimensionless(u2, D / 2, U0).T,
                     extent=[G2_X1 / D, G2_X2 / D,
                             G2_Y1 / D, G2_Y2 / D], **kwargs)
    im3 = plt.imshow(post.calculate_vorticity_dimensionless(u3, D, U0).T,
                     extent=[G3_X1 / D, G3_X2 / D, 
                             G3_Y1 / D, G3_Y2 / D], **kwargs)
    plt.colorbar(label="Vorticity")
    
    # Draw cylinder
    cylinder = plt.Circle(((OBJ_X + d[0]) / D, (OBJ_Y + d[1]) / D), 0.5, 
                          edgecolor='black', linewidth=0.5, facecolor='white', fill=True)
    plt.gca().add_artist(cylinder)
    
    # Mark initial position of cylinder
    plt.plot(OBJ_X / D, OBJ_Y / D, marker='+', markersize=10, 
             color='k', linestyle='None', markeredgewidth=0.5)
    
    plt.xlabel("x / D")
    plt.ylabel("y / D")
    plt.tight_layout()


# ========================== RUN SIMULATION ==========================

for t in tqdm(range(TM)):
    (
        f1, rho1, u1, 
        f2, rho2, u2,
        f3, rho3, u3,
        d, v, a, h,
    ) = update_grid1(f1, f2, f3, d, v, a, h)
    
    # Store history
    d_hist[:, t] = np.array(d)
    h_hist[:, t] = np.array(h)
    
    # Update visualization
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
        
        # Update vorticity fields
        im1.set_data(post.calculate_vorticity_dimensionless(u1, D / 4, U0).T)
        im2.set_data(post.calculate_vorticity_dimensionless(u2, D / 2, U0).T)
        im3.set_data(post.calculate_vorticity_dimensionless(u3, D, U0).T)

        # Update cylinder position
        obj_x = (OBJ_X + d[0]) / D
        obj_y = (OBJ_Y + d[1]) / D
        cylinder.set_center((obj_x, obj_y))
        
        # Update IB region position
        ib_region.set_xy(((G3_X1 + IB_X1_ + d[0]).astype(jnp.int32) / D, 
                          (G3_Y1 + IB_Y1_ + d[1]).astype(jnp.int32) / D))
        ib_label.set_position((obj_x, obj_y + 0.6))
        
        plt.pause(0.01)
