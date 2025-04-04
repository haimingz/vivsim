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

from vivsim import ib, lbm, multigrid as mg, post, mrt


# ========================== CONSTANTS =======================

D = 20                 # Cylinder diameter
NY = 20 * D            # Height of the domain
NX = 40 * D            # Width of the domain

# Cylinder position
X_OBJ = 10 * D           # Cylinder x position
Y_OBJ = 10 * D          # Cylinder y position

# IB method parameters
N_MARKER = 4 * D       # Number of markers on cylinder
N_ITER_MDF = 5          # Multi-direct forcing iterations
IB_PADDING = 2         # Padding around the cylinder

# Physical parameters
U0 = 0.02              # Inlet velocity
RE = 100              # Reynolds number
TM = int(30 * 5 / U0 * D)             # Total time steps

# ==================== DERIVED CONSTANTS =======================

# fluid parameter
NU = U0 * D / RE                                            # Kinematic viscosity
OMEGA = 1 / (3 * NU + 0.5)                                  # Lattice Boltzmann relaxation time
MRT_COL_LEFT, MRT_SRC_LEFT = mrt.precompute_left_matrices(OMEGA)

# IBM parameters
THETA_MAKERS = jnp.linspace(0, jnp.pi * 2, N_MARKER, dtype=jnp.float32, endpoint=False)
X_MARKERS = X_OBJ + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS = Y_OBJ + 0.5 * D * jnp.sin(THETA_MAKERS)
L_ARC = D * math.pi / N_MARKER  # arc length between the markers

# dynamic ibm region
IB_IDX1 = int(X_OBJ - 0.5 * D - IB_PADDING)
IB_IDY1 = int(Y_OBJ - 0.5 * D - IB_PADDING)
IB_SIZE = D + IB_PADDING * 2

# coord of background grid
X, Y = jnp.mgrid[0:NX, 0:NY]

# ======================= define variables =====================

f, rho, u = mg.init_grid(NX, NY)
d = jnp.zeros((2), dtype=jnp.float32) 
v = jnp.zeros((2), dtype=jnp.float32) 
a = jnp.zeros((2), dtype=jnp.float32) 
h = jnp.zeros((2), dtype=jnp.float32) 

# ======================= initial conditions =====================

u = u.at[0].set(U0)
f =  lbm.get_equilibrium(rho, u)
feq_init = f[:, 1, 1]


# ======================= compute routine =====================

@jax.jit
def update(f, d, v, a, h):

    # update macroscopic and collision
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = mrt.collision(f, feq, MRT_COL_LEFT)
    
    # extract data from ibm region
    u_slice = u[:, IB_IDX1:IB_IDX1+IB_SIZE, IB_IDY1:IB_IDY1+IB_SIZE]
    X_slice = X[IB_IDX1:IB_IDX1+IB_SIZE, IB_IDY1:IB_IDY1+IB_SIZE]
    Y_slice = Y[IB_IDX1:IB_IDX1+IB_SIZE, IB_IDY1:IB_IDY1+IB_SIZE]
    
    # calculate ibm force
    g_slice, h_markers = ib.multi_direct_forcing(u_slice, X_slice, Y_slice, 
                                           v, X_MARKERS, Y_MARKERS, N_MARKER, L_ARC, 
                                           N_ITER_MDF, ib.kernel_range4)
    
    # apply the force to the fluid
    g_slice_discrete = lbm.get_discretized_force(g_slice, u_slice)
    s_slice = mrt.get_source(g_slice_discrete, MRT_SRC_LEFT)    
    f = f.at[:, IB_IDX1:IB_IDX1+IB_SIZE, IB_IDY1:IB_IDY1+IB_SIZE].add(s_slice)

    # calculate the total force to the cylinder
    h = ib.get_force_to_obj(h_markers)
    
    # streaming and applying boundary conditions
    f = lbm.streaming(f)
    f = lbm.velocity_boundary(f, U0, 0, loc='left')
    f = lbm.boundary_equilibrium(f, feq_init[:, jnp.newaxis], loc='right') 
       
    return f, rho, u, d, v, a, h


# ======================= create plot template =====================

mpl.rcParams['figure.raise_window'] = False

# first plot
curl = post.calculate_curl(u)

fig1, ax1 = plt.subplots(figsize=(4, 2))
im = ax1.imshow(
    curl.T,
    extent=[0, NX/D, 0, NY/D],
    cmap="Blues",
    aspect="equal",
    origin="lower",
)
ax1.set_xlabel("x/D")
ax1.set_ylabel("y/D")
fig1.colorbar(im, ax=ax1)
im.set_clim(0, U0 * 1.5)
fig1.tight_layout()


# second plot
h_history = jnp.zeros((2, TM), dtype=jnp.float32)
cd = h_history[0] * 2 / (D * U0 ** 2)
cl = h_history[1] * 2 / (D * U0 ** 2)
cd_mean = jnp.mean(cd[TM//2:])
cl_max = jnp.max(cl[TM//2:])

fig2, ax2 = plt.subplots(figsize=(6, 2))
l1, = ax2.plot(cd, label='Cd')
l2, = ax2.plot(cl, label='Cl')
ax2.set_xlabel("t")
ax2.legend(frameon=False)
ax2.set_ylim(-2, 4)
text = ax2.text(0.5, 0.9, f"", 
                       transform=ax2.transAxes,  ha="center", va="center", 
                       fontsize=8, color='blue')
fig2.tight_layout()


# ========================== start simulation ==========================

for t in tqdm(range(TM)):
    f, rho, u, d, v, a, h = update(f, d, v, a, h)
    h_history = h_history.at[:, t].set(h)
    
    if t % 400 == 0:
        im.set_data(post.calculate_velocity(u).T)
        fig1.canvas.draw()
        fig1.canvas.flush_events()
        
        cl = h_history[1] * 2 / (D * U0 ** 2)
        cd = h_history[0] * 2 / (D * U0 ** 2)         
        l1.set_ydata(cl)
        l2.set_ydata(cd)
        
        if t > TM//2:
            cd_mean = jnp.mean(cd[TM//2:t])
            cl_max = jnp.max(cl[TM//2:t])
            text.set_text(f"Cd_mean={cd_mean:.3f}, Cl_max={cl_max:.3f}")
        

        fig2.canvas.draw()
        fig2.canvas.flush_events()        
        plt.pause(0.01)

plt.show()
