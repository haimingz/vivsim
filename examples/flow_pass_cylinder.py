"""
Flow past a stationary cylinder using IB-LBM

This example can be used to validate the IB-LBM implementation via comparison the 
hydrodynamic force coefficients against reference data from literature.
"""

import math

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

from vivsim import dyn, ib, lbm


# ========================== GEOMETRY =======================

# Cylinder geometry
D = 60                      # Cylinder diameter
NY = 20 * D                 # Domain height
NX = 30 * D                 # Domain width
X_CYL = 10 * D              # Cylinder center x-position
Y_CYL = 10 * D              # Cylinder center y-position
N_MARKER = 4 * D            # Number of markers on cylinder surface


# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.05                   # Inlet velocity
RE = 200                    # Reynolds number
NU = U0 * D / RE            # Kinematic viscosity
TM = int(30 * 5 / U0 * D)   # Total simulation timesteps


# ========================== IB-LBM PARAMETERS ==========================

OMEGA = lbm.get_omega(NU)   # Relaxation parameter
N_ITER = 3                  # Multi-direct forcing iterations
IB_PAD = 2                  # Padding around cylinder for IBM region


# ========================== REF DATA ==========================

# Reference data, averaging the results from the following sources:
# [1] https://doi.org/10.1016/j.ijmecsci.2024.108961.
# [2] https://doi.org/10.1016/j.camwa.2014.05.013.
# [3] https://doi.org/10.1017/S0022112086003014

CD_REF = 1.389              # Mean drag coefficient
CL_REF = 0.718              # Maximum lift coefficient amplitude


# ==================== MESHES ========================

# LBM grid coordinates
X, Y = jnp.mgrid[0:NX, 0:NY]

# IBM marker configuration
THETA_MARKERS = jnp.linspace(0, 2 * jnp.pi, N_MARKER, endpoint=False)
X_MARKERS = X_CYL + 0.5 * D * jnp.cos(THETA_MARKERS)
Y_MARKERS = Y_CYL + 0.5 * D * jnp.sin(THETA_MARKERS)
DS_MARKERS = D * math.pi / N_MARKER

# IBM region (for optimized computation)
IB_X0 = int(X_CYL - 0.5 * D - IB_PAD)
IB_Y0 = int(Y_CYL - 0.5 * D - IB_PAD)
IB_SIZE = D + 2 * IB_PAD
IB_REGION = (slice(IB_X0, IB_X0 + IB_SIZE), slice(IB_Y0, IB_Y0 + IB_SIZE))

# Pre-compute IBM region coordinates and kernels
X_IB = X[IB_REGION]
Y_IB = Y[IB_REGION]
KERNELS = ib.get_kernels(X_MARKERS, Y_MARKERS, X_IB, Y_IB, ib.kernel_range4)


# ======================= INITIALIZE VARIABLES ====================

rho = jnp.ones((NX, NY))
u = jnp.zeros((2, NX, NY))
u = u.at[0].set(U0)
f = lbm.get_equilibrium(rho, u)
h = jnp.zeros(2) 


# ======================= SIMULATION ROUTINE =====================

@jax.jit
def update(f, h):
    
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)
    
    # Extract IBM region data for efficient computation
    u_ib = u[:, *IB_REGION]
    f_ib = f[:, *IB_REGION]
    rho_ib = rho[IB_REGION]
    
    # Compute IBM forcing using multi-direct forcing method
    g_ib, h_marker = ib.multi_direct_forcing(u_ib, 0, KERNELS, DS_MARKERS, n_iter=N_ITER)
    
    # Integrate marker forces to get total force on cylinder
    h = dyn.get_force_to_obj(h_marker)
    
    # Update distribution functions and velocity in IBM region
    f = f.at[:, *IB_REGION].set(lbm.forcing_edm(f_ib, g_ib, u_ib, rho_ib))
    u = u.at[:, *IB_REGION].add(lbm.get_velocity_correction(g_ib, rho_ib))

    f = lbm.streaming(f)
    f = lbm.boundary_nebb(f, loc='left', ux_wall=U0)          
    f = lbm.boundary_equilibrium(f, loc='right', ux_wall=U0)  
    
    return f, rho, u, h


# ======================= VISUALIZATION SETUP ====================

mpl.rcParams['figure.raise_window'] = False

# Storage for force history
h_hist = jnp.zeros((2, TM), dtype=jnp.float32)


# ========================== RUN SIMULATION ==========================

for t in tqdm(range(TM)):
    f, rho, u, h = update(f, h)
    h_hist = h_hist.at[:, t].set(h)


# ======================= PLOT RESULTS ==========================

# Compute normalized force coefficients
cd = h_hist[0] * 2 / (D * U0**2)
cl = h_hist[1] * 2 / (D * U0**2)

# Compute statistics from second half of simulation
cd_mean = np.mean(cd[TM // 2:])
cl_max = np.max(np.abs(cl[TM // 2:]))
cd_err = abs(cd_mean - CD_REF) / CD_REF * 100
cl_err = abs(cl_max - CL_REF) / CL_REF * 100

# Create plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(cd, label="Cd (drag)", linewidth=1.5)
ax.plot(cl, label="Cl (lift)", linewidth=1.5)

ax.set_xlabel("Time step", fontsize=10)
ax.set_ylabel("Force coefficient", fontsize=10)
ax.legend(frameon=False, loc='upper right', fontsize=9)
ax.set_ylim(-2, 4)
ax.grid(True, alpha=0.3, linestyle=':')

# Add statistics text
stats_text = (
    f"Cd = {cd_mean:.3f} (ref: {CD_REF}, err: {cd_err:.1f}%)\n"
    f"Cl = {cl_max:.3f} (ref: {CL_REF}, err: {cl_err:.1f}%)"
)
ax.text(
    0.02, 0.95, stats_text,
    transform=ax.transAxes,
    ha="left", va="top",
    fontsize=9, color="blue",
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

fig.tight_layout()
plt.show()

