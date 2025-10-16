"""
Vortex-Induced Vibration (VIV) of a circular cylinder using IB-LBM

This example simulates the vortex-induced vibration of a circular cylinder
in a 2D flow using the Immersed Boundary Method (IBM) coupled with the
Lattice Boltzmann Method (LBM). The cylinder is free to oscillate in both
cross-flow and in-line directions with spring-mass-damper constraints.
"""

import math

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

from vivsim import dyn, ib, lbm, post



# ========================== VISUALIZATION PARAMETERS ====================

PLOT = True                 # Enable real-time visualization
PLOT_EVERY = 200            # Plot update frequency
PLOT_AFTER = 0              # Start plotting after this timestep


# ========================== GEOMETRY =======================

D = 32                      # Cylinder diameter
NX = 20 * D                 # Domain width
NY = 10 * D                 # Domain height
X_CYL = 5 * D               # Cylinder center x-position
Y_CYL = 5 * D               # Cylinder center y-position
N_MARKER = 5 * D            # Number of markers on cylinder surface


# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.05                   # Inlet velocity
RE = 150                    # Reynolds number
UR = 5                      # Reduced velocity
MR = 10                     # Mass ratio
DR = 0                      # Damping ratio
NU = U0 * D / RE            # Kinematic viscosity
FN = U0 / (UR * D)                                  # Natural frequency
M = math.pi * (D / 2) ** 2 * MR                     # Mass of the cylinder
K = (FN * 2 * math.pi) ** 2 * M * (1 + 1 / MR)      # Spring stiffness
C = 2 * math.sqrt(K * M) * DR                       # Damping coefficient
TM = int(30 / FN)                                   # Total simulation timesteps


# ========================== IB-LBM PARAMETERS ==========================

OMEGA = lbm.get_omega(NU)    # Relaxation parameter
IB_ITER = 3                  # Multi-direct forcing iterations
IB_PAD = 10                  # Padding around cylinder for IBM region
FSI_ITER = 5                 # Number of FSI sub-iterations per time step


# ==================== MESHES ========================

# LBM grid coordinates
X, Y = jnp.mgrid[0:NX, 0:NY]

# IBM marker configuration
THETA = jnp.linspace(0, 2 * jnp.pi, N_MARKER, endpoint=False)
X_MARKERS = X_CYL + 0.5 * D * jnp.cos(THETA)
Y_MARKERS = Y_CYL + 0.5 * D * jnp.sin(THETA)
DS_MARKERS = D * math.pi / N_MARKER

IB_X0 = int(X_CYL - 0.5 * D - IB_PAD)
IB_Y0 = int(Y_CYL - 0.5 * D - IB_PAD)
IB_SIZE = D + 2 * IB_PAD


# ======================= INITIALIZE VARIABLES ====================

# Fluid variables
rho = jnp.ones((NX, NY))
u = jnp.zeros((2, NX, NY))
u = u.at[0].set(U0)
f = lbm.get_equilibrium(rho, u)

# Structural variables
d = jnp.zeros(2)            # Displacement of cylinder
v = jnp.zeros(2)            # Velocity of cylinder
a = jnp.zeros(2)            # Acceleration of cylinder
h = jnp.zeros(2)            # Hydrodynamic force

# Optional: add initial perturbation
v = v.at[1].set(1e-2 * U0)


# ======================= SIMULATION ROUTINE =====================

@jax.jit
def update(f, rho, u, d, v, a, h):
    
    # Update macroscopic variables
    rho, u = lbm.get_macroscopic(f)
    
    # Collision step
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)
    
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
        
        x_markers, y_markers = dyn.get_markers_coords_2dof(X_MARKERS, Y_MARKERS, d)
        kernels = ib.get_kernels(x_markers, y_markers, x_ib, y_ib, ib.kernel_range4)
        g_ib, h_marker = ib.multi_direct_forcing(u_ib, v, kernels, DS_MARKERS, n_iter=IB_ITER)
        
        # apply marker forces to the cylinder
        h = dyn.get_force_to_obj(h_marker)
        h += a * math.pi * D ** 2 / 4  # internal mass correction
        a, v, d = dyn.newmark_2dof(a_old, v_old, d_old, h, M, K, C)
    
    # Update distribution functions and velocity in IBM region
    f_ib = lbm.forcing_edm(f_ib, g_ib, u_ib, rho_ib)
    f = jax.lax.dynamic_update_slice(f, f_ib, (0, ib_x0, ib_y0))
    
    u_ib = u_ib + lbm.get_velocity_correction(g_ib, rho_ib)
    u = jax.lax.dynamic_update_slice(u, u_ib, (0, ib_x0, ib_y0))
    
    # Streaming step
    f = lbm.streaming(f)
    
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
    
    plt.colorbar(label="Vorticity * D / U0")
    plt.xlabel("x / D")
    plt.ylabel("y / D")
    
    # Mark the initial position of the cylinder
    plt.plot(X_CYL / D, Y_CYL / D, marker='+', markersize=10, 
             color='k', linestyle='None', markeredgewidth=0.5)
    
    plt.tight_layout()


# ========================== RUN SIMULATION ==========================

for t in tqdm(range(TM)):
    f, rho, u, d, v, a, h = update(f, rho, u, d, v, a, h)
    
    # Store history
    d_hist[:, t] = np.array(d)
    h_hist[:, t] = np.array(h)
    
    # Update visualization
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
        curl = post.calculate_vorticity_dimensionless(u, D, U0)
        im.set_data(curl.T)
        plt.pause(0.001)


# ======================= POST-PROCESSING ==========================

# Plot displacement and force time histories
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

t_normalized = np.arange(TM) *  FN

ax1.plot(t_normalized, d_hist[0] / D, label='x / D', linewidth=1)
ax1.plot(t_normalized, d_hist[1] / D, label='y / D', linewidth=1)
ax1.set_xlabel('Time step')
ax1.set_ylabel('Displacement / D')
ax1.legend(loc='upper left', ncol=2)

cd = h_hist[0] * 2 / (D * U0**2)
cl = h_hist[1] * 2 / (D * U0**2)

ax2.plot(t_normalized, cd, label='Cd (drag)', linewidth=1)
ax2.plot(t_normalized, cl, label='Cl (lift)', linewidth=1)
ax2.set_xlabel('Time step')
ax2.set_ylabel('Force coefficient')
ax2.legend(loc='upper left', ncol=2)
ax2.set_ylim(-2, 4)

fig.tight_layout()
plt.show()
