"""
Vortex-Induced Vibration (VIV) with Multi-Block Grid System

This example simulates the vortex-induced vibration of a circular cylinder
in a 2D flow using the Immersed Boundary Method (IBM) coupled with the
Lattice Boltzmann Method (LBM) with a multi-block grid system. The cylinder
is free to oscillate in both cross-flow and in-line directions with 
spring-mass-damper constraints. The simulation uses five blocks with
different levels of refinement for computational efficiency.
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
PLOT_EVERY = 100            # Plot update frequency
PLOT_AFTER = 0              # Start plotting after this timestep


# ========================== PHYSICAL PARAMETERS =====================

D = 48                      # Cylinder diameter
RE = 500                     # Reynolds number
UR = 5                      # Reduced velocity
MR = 10                     # Mass ratio
DR = 0                      # Damping ratio
U0 = 0.05                                            # Inlet velocity
NU = U0 * D / RE                                    # Kinematic viscosity
FN = U0 / (UR * D)                                  # Natural frequency
AREA = math.pi * (D / 2) ** 2                       # Cylinder cross-sectional area
M = AREA * MR                                       # Mass of the cylinder
K = (FN * 2 * math.pi) ** 2 * M * (1 + 1 / MR)      # Spring stiffness
C = 2 * math.sqrt(K * M) * DR                       # Damping coefficient
TM = 60000                                          # Total simulation timesteps


# ========================== GEOMETRY PARAMETERS =====================


HEIGHT = 20 * D             # Domain height (constant across all blocks)
WIDTHS = jnp.array([8, 0.5, 3, 0.5, 18]) * D   # Width of each block
LEVELS = [-2, -1, 0, -1, -2]                        # Refinement level of each block
X_CYL = 10 * D - WIDTHS[0] - WIDTHS[1]
Y_CYL = 10 * D   

# Block 2 coordinates for IB calculations
X, Y = jnp.meshgrid(jnp.arange(WIDTHS[2]), jnp.arange(HEIGHT), indexing='ij')

# IBM marker configuration
N_MARKER = 4 * D            # Number of markers on cylinder surface
THETA_MARKERS = jnp.linspace(0, 2 * jnp.pi, N_MARKER, dtype=jnp.float32, endpoint=False)
X_MARKERS = X_CYL + 0.5 * D * jnp.cos(THETA_MARKERS)  # Initial x positions (relative to Block 2)
Y_MARKERS = Y_CYL + 0.5 * D * jnp.sin(THETA_MARKERS)  # Initial y positions (relative to Block 2)
DS_MARKERS = D * math.pi / N_MARKER                    # Arc length between markers

# IBM region configuration
IB_PAD = 10                                          # Padding around cylinder for IBM region
IB_X0 = int(X_CYL - 0.5 * D - IB_PAD)   # Initial x position of IB region
IB_Y0 = int(Y_CYL - 0.5 * D - IB_PAD)   # Initial y position of IB region
IB_SIZE = D + IB_PAD * 2                            # Size of IB region

IB_ITER = 3                                         # Multi-direct forcing iterations
FSI_ITER = 1                                        # FSI sub-iterations per time step

# Relaxation parameters for each block
OMEGA0 = mg.get_omega(NU, LEVELS[0])    # Block 0 relaxation parameter
OMEGA1 = mg.get_omega(NU, LEVELS[1])    # Block 1 relaxation parameter
OMEGA2 = mg.get_omega(NU, LEVELS[2])    # Block 2 relaxation parameter (finest)
OMEGA3 = mg.get_omega(NU, LEVELS[3])    # Block 3 relaxation parameter
OMEGA4 = mg.get_omega(NU, LEVELS[4])    # Block 4 relaxation parameter


# ======================= INITIALIZE VARIABLES ====================

# Initialize grids for all blocks
f0, rho0, u0 = mg.init_grid(WIDTHS[0], HEIGHT, LEVELS[0], buffer_x=1)
f1, rho1, u1 = mg.init_grid(WIDTHS[1], HEIGHT, LEVELS[1], buffer_x=1)
f2, rho2, u2 = mg.init_grid(WIDTHS[2], HEIGHT, LEVELS[2])
f3, rho3, u3 = mg.init_grid(WIDTHS[3], HEIGHT, LEVELS[3], buffer_x=1)
f4, rho4, u4 = mg.init_grid(WIDTHS[4], HEIGHT, LEVELS[4], buffer_x=1)

# Set initial velocity
u0 = u0.at[0].set(U0)
u1 = u1.at[0].set(U0)
u2 = u2.at[0].set(U0)
u3 = u3.at[0].set(U0)
u4 = u4.at[0].set(U0)

# Initialize distribution functions
f0 = lbm.get_equilibrium(rho0, u0)
f1 = lbm.get_equilibrium(rho1, u1)
f2 = lbm.get_equilibrium(rho2, u2)
f3 = lbm.get_equilibrium(rho3, u3)
f4 = lbm.get_equilibrium(rho4, u4)

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
    ib_x0 = (IB_X0 + d[0]).astype(jnp.int32)
    ib_y0 = (IB_Y0 + d[1]).astype(jnp.int32)
    
    # Extract data from IBM region for efficient computation
    rho_ib = jax.lax.dynamic_slice(rho, (ib_x0, ib_y0), (IB_SIZE, IB_SIZE))
    u_ib = jax.lax.dynamic_slice(u, (0, ib_x0, ib_y0), (2, IB_SIZE, IB_SIZE))
    x_ib = jax.lax.dynamic_slice(X, (ib_x0, ib_y0), (IB_SIZE, IB_SIZE))
    y_ib = jax.lax.dynamic_slice(Y, (ib_x0, ib_y0), (IB_SIZE, IB_SIZE))
    f_ib = jax.lax.dynamic_slice(f, (0, ib_x0, ib_y0), (9, IB_SIZE, IB_SIZE))
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
    f = jax.lax.dynamic_update_slice(f, f_ib, (0, ib_x0, ib_y0))
    
    u_ib = u_ib + lbm.get_velocity_correction(g_ib, rho_ib)
    u = jax.lax.dynamic_update_slice(u, u_ib, (0, ib_x0, ib_y0))
    
    return f, u, d, v, a, h


def update_block2(f1, f2, f3, d, v, a, h):
    """Update Block 2 (finest grid) with FSI coupling."""
    
    # Macroscopic variables and collision
    rho2, u2 = lbm.get_macroscopic(f2)
    feq2 = lbm.get_equilibrium(rho2, u2)
    f2 = lbm.collision_kbc(f2, feq2, OMEGA2)
    
    # Solve FSI
    f2, u2, d, v, a, h = solve_fsi(f2, rho2, u2, d, v, a, h)
    
    # Streaming step
    f2 = lbm.streaming(f2)
    
    # Outer boundary - interpolation from adjacent blocks
    f2 = mg.coarse_to_fine(f1, f2, dir='right')
    f2 = mg.coarse_to_fine(f3, f2, dir='left')
    
    return f2, rho2, u2, d, v, a, h


def update_level1(f1, f2, f3, f4, d, v, a, h):
    """Update Level 1 blocks (Blocks 1 and 3)."""
    
    # Macroscopic variables and collision for Blocks 1 and 3
    rho1, u1 = lbm.get_macroscopic(f1)
    feq1 = lbm.get_equilibrium(rho1, u1)
    f1 = lbm.collision_kbc(f1, feq1, OMEGA1)
    
    rho3, u3 = lbm.get_macroscopic(f3)
    feq3 = lbm.get_equilibrium(rho3, u3)
    f3 = lbm.collision_kbc(f3, feq3, OMEGA3)
    
    # Streaming step
    f1 = lbm.streaming(f1)
    f3 = lbm.streaming(f3)
    
    # Update finest block (Block 2) twice per Level 1 update
    f2, rho2, u2, d, v, a, h = update_block2(f1, f2, f3, d, v, a, h)
    f2, rho2, u2, d, v, a, h = update_block2(f1, f2, f3, d, v, a, h)
    
    # Update Block 1 and 3 boundaries with Block 2 data
    f1 = mg.fine_to_coarse(f2, f1, dir='left')
    f1 = mg.coarse_to_fine(f0, f1, dir='right')
    
    f3 = mg.fine_to_coarse(f2, f3, dir='right')
    f3 = mg.coarse_to_fine(f4, f3, dir='left')    
    
    return (f1, rho1, u1, 
            f2, rho2, u2,
            f3, rho3, u3, 
            d, v, a, h)


@jax.jit
def update_level2(f0, f1, f2, f3, f4, d, v, a, h):
    """Update Level 2 blocks (Blocks 0 and 4) and manage nested updates."""
    
    # Macroscopic variables and collision for Blocks 0 and 4
    rho0, u0 = lbm.get_macroscopic(f0)
    feq0 = lbm.get_equilibrium(rho0, u0)
    f0 = lbm.collision_kbc(f0, feq0, OMEGA0)
    
    rho4, u4 = lbm.get_macroscopic(f4)
    feq4 = lbm.get_equilibrium(rho4, u4)
    f4 = lbm.collision_kbc(f4, feq4, OMEGA4)
    
    # Streaming step
    f0 = lbm.streaming(f0)
    f4 = lbm.streaming(f4)
    
    # Update finer blocks (Blocks 1, 2, 3) twice per Level 2 update
    (f1, rho1, u1, 
     f2, rho2, u2,
     f3, rho3, u3, 
     d, v, a, h) = update_level1(f1, f2, f3, f4, d, v, a, h)
    (f1, rho1, u1, 
     f2, rho2, u2,
     f3, rho3, u3, 
     d, v, a, h) = update_level1(f1, f2, f3, f4, d, v, a, h)

    # Update Block 0 and 4 boundaries
    f0 = mg.fine_to_coarse(f1, f0, dir='left')
    f0 = lbm.boundary_nee(f0, loc='left', ux_wall=U0)
    
    f4 = mg.fine_to_coarse(f3, f4, dir='right') 
    f4 = lbm.boundary_equilibrium(f4, loc='right', ux_wall=U0)
    
    return (f0, rho0, u0,
            f1, rho1, u1, 
            f2, rho2, u2, 
            f3, rho3, u3, 
            f4, rho4, u4, 
            d, v, a, h)


# ======================= VISUALIZATION SETUP ====================

mpl.rcParams['figure.raise_window'] = False

# Storage for displacement and force history
d_hist = np.zeros((2, TM))
h_hist = np.zeros((2, TM))

# Setup real-time visualization
if PLOT:
    
    plt.figure(figsize=(10, 4))

    kwargs = dict(
        cmap="seismic", 
        aspect="equal", 
        origin="lower",
        vmax=0.01,
        vmin=-0.01,
    )

    # Initialize vorticity plots for each block
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
        extent=[jnp.sum(WIDTHS[:3]) / D, jnp.sum(WIDTHS[:4]) / D, 0, HEIGHT / D],
        **kwargs
    )

    im4 = plt.imshow(
        post.calculate_curl(u4).T,
        extent=[jnp.sum(WIDTHS[:4]) / D, jnp.sum(WIDTHS) / D, 0, HEIGHT / D],
        **kwargs
    )

    plt.colorbar(label="Vorticity")
    plt.xlabel("x / D")
    plt.ylabel("y / D")
    
    # Draw block boundaries
    plt.axvline(jnp.sum(WIDTHS[:1]) / D, color="g", linestyle="--", linewidth=0.5)
    plt.axvline(jnp.sum(WIDTHS[:2]) / D, color="g", linestyle="--", linewidth=0.5)   
    plt.axvline(jnp.sum(WIDTHS[:3]) / D, color="g", linestyle="--", linewidth=0.5)
    plt.axvline(jnp.sum(WIDTHS[:4]) / D, color="g", linestyle="--", linewidth=0.5)

    # Mark initial position of cylinder
    plt.plot((X_CYL + WIDTHS[0] + WIDTHS[1]) / D, Y_CYL / D, 
             marker='+', markersize=10, color='k', linestyle='None', markeredgewidth=0.5)
    
    plt.tight_layout()



# ========================== RUN SIMULATION ==========================

for t in tqdm(range(TM)):
    (
        f0, rho0, u0, 
        f1, rho1, u1, 
        f2, rho2, u2, 
        f3, rho3, u3, 
        f4, rho4, u4, 
        d, v, a, h
    ) = update_level2(f0, f1, f2, f3, f4, d, v, a, h)
    
    # Store history
    d_hist[:, t] = np.array(d)
    h_hist[:, t] = np.array(h)
    
    # Update visualization
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:

        im0.set_data(post.calculate_curl(u0).T / D / U0 / 4)
        im1.set_data(post.calculate_curl(u1).T / D / U0 / 2)
        im2.set_data(post.calculate_curl(u2).T / D / U0 / 1)
        im3.set_data(post.calculate_curl(u3).T / D / U0 / 2)   
        im4.set_data(post.calculate_curl(u4).T / D / U0 / 4)  
        plt.pause(0.01)