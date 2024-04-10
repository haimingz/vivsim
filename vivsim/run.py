import jax
import jax.numpy as jnp
from tqdm import tqdm
import math
import os
import json
from iblbm import core, dynamics, post
import matplotlib.pyplot as plt
import matplotlib as mpl


# ----------------- read parameters from json file -----------------

working_dir = os.path.dirname(os.path.abspath(__file__))
with open(working_dir + "/parameters.json") as f:
    params = json.load(f)

D = params["D"]
U0 = params["U"]
M = params["M"]
K = params["K"]
C = params["C"]
NU = params["NU"]
NX = params["NX"]
NY = params["NY"]
TM = params["TM"]
N_MARKER = params["N_MARKER"]
X_OBJ = params["X_CYLINDER"]
Y_OBJ = params["Y_CYLINDER"]

# ----------------- LBM parameters -----------------

TAU = 3 * NU + 0.5  # relaxation time
OMEGA = 1 / TAU  # relaxation parameter
MRT_OMEGA = core.get_omega_mrt(OMEGA)   # relaxation matrix for MRT
L_ARC = D * math.pi / N_MARKER  # arc length between the markers
X1 = int(X_OBJ - 0.7 * D)   # left boundary of the IBM region
X2 = int(X_OBJ + 1.0 * D)   # right boundary of the IBM region
Y1 = int(Y_OBJ - 1.5 * D)   # bottom boundary of the IBM region
Y2 = int(Y_OBJ + 1.5 * D)   # top boundary of the IBM region
MDF_ITER = 1  # number of iterations for multi-direct forcing


# ----------------- output parameters -----------------
PLOT = True
PLOT_EVERY = 50
PLOT_AFTER = 00
N_PLOTS = int((TM - PLOT_AFTER) // PLOT_EVERY)
PLOT_CONTENT = "curl"

# ----------------- define properties -----------------

# mesh grid
X, Y = jnp.meshgrid(jnp.arange(NX, dtype=jnp.int16), 
                    jnp.arange(NY, dtype=jnp.int16), indexing="ij")

THETA_MAKERS = jnp.linspace(0, jnp.pi * 2, N_MARKER, endpoint=False)
X_MARKERS = X_OBJ + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS = Y_OBJ + 0.5 * D * jnp.sin(THETA_MAKERS)


# macroscopic properties
rho = jnp.ones((NX, NY), dtype=jnp.float32)  # density of fluid
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)  # velocity of fluid

# microscopic properties
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # distribution functions
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # equilibrium distribution functions

# structural dynamics properties
d = jnp.zeros((2), dtype=jnp.float32)  # displacement
v = jnp.zeros((2), dtype=jnp.float32)  # velocity
a = jnp.zeros((2), dtype=jnp.float32)  # acceleration
g = jnp.zeros((2), dtype=jnp.float32)  # force


# initilize properties
# u = u.at[0].set(U0)
f = core.equilibrum(rho, u, f)


# ----------------- define main loop -----------------
@jax.jit
def update(f, feq, rho, u, d, v, a, g):
    """Update for one time step"""

    # Immersed Boundary Method
    g_to_markers = jnp.zeros((N_MARKER, 2))  # force to the markers
    g_to_fluid = jnp.zeros((2, X2 - X1, Y2 - Y1))  # force to the fluid
    
    for _ in range(MDF_ITER):
        
        # calculate the kernels
        x_markers = X_MARKERS + d[0]  # x coordinates of the markers
        y_markers = Y_MARKERS + d[1]  # y coordinates of the markers
        kernels = jax.vmap(core.kernel3, in_axes=(0, 0, None, None))(x_markers, y_markers, X[X1:X2, Y1:Y2], Y[X1:X2, Y1:Y2])
        
        # interpolate velocity at the markers
        u_markers = jax.vmap(core.interpolate_u, in_axes=(None, 0))(u[:, X1:X2, Y1:Y2], kernels)
        
        # calculate and apply the needed correction force to the fluid
        g_needed = jax.vmap(core.get_g_correction, in_axes=(None, 0))(v, u_markers)
        g_needed_spread = jnp.sum(jax.vmap(core.spread_g, in_axes=(0, 0))(g_needed, kernels), axis=0)
        u = u.at[:, X1:X2, Y1:Y2].add(core.get_u_correction(g_needed_spread))
        
        # accumulate the coresponding correction force to the markers and the fluid
        g_to_markers += - g_needed
        g_to_fluid += g_needed_spread

    # Compute solid dynamics
    g = jnp.sum(g_to_markers, axis=0) * L_ARC
    a, v, d = dynamics.newmark(a, v, d, g, M, K, C)

    # Compute equilibrium
    feq = core.equilibrum(rho, u, feq)

    # Collision
    f = core.collision_mrt(f, feq, MRT_OMEGA)
    
    # Add source term
    f = f.at[:, X1:X2, Y1:Y2].add(core.get_source(u[:, X1:X2, Y1:Y2], g_to_fluid, OMEGA))

    # Streaming
    f = core.streaming(f)

    # Set Outlet BC at right wall (No gradient BC)
    f = core.right_outlet(f)

    # Set Inlet BC at left wall (Non-equilibrium Bounce-Back)
    f, rho, u = core.left_inlet(f, rho, u, U0)

    # update new macroscopic
    rho, u = core.get_macroscopic(f, rho, u)
     
    return f, feq, rho, u, d, v, a, g

# ----------------- start simulation -----------------

if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    plt.figure(figsize=(8, 4))
    
for t in tqdm(range(TM)):
    f, feq, rho, u, d, v, a, g = update(f, feq, rho, u, d, v, a, g)

    
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
        plt.clf()

        if PLOT_CONTENT == "curl":

            curl = post.calculate_curl(u)
            plt.imshow(
                curl.T,
                extent=[0, NX/D, 0, NY/D],
                cmap="seismic",
                aspect="equal",
                # norm=mpl.colors.CenteredNorm(),
                vmax=0.03,
                vmin=-0.03
            )
        
        if PLOT_CONTENT == "rho":
            plt.imshow(
                rho.T,
                extent=[0, NX/D, 0, NY/D],
                cmap="seismic",
                aspect="equal",
                # norm=mpl.colors.CenteredNorm(vcenter=1),
                vmin=0.95,
                vmax=1.05
            )
        
        plt.colorbar()
        
        # plt.xticks([])
        # plt.yticks([])
        plt.xlabel("x/D")
        plt.ylabel("y/D")

        # draw the central lines
        # plt.axvline(X_OBJ / D, color="k", linestyle="--", linewidth=0.5)
        # plt.axhline(Y_OBJ / D, color="k", linestyle="--", linewidth=0.5)
        
        # draw outline of the IBM region as a rectangle
        # plt.plot([X1, X1, X2, X2, X1], 
        #          [Y1, Y2, Y2, Y1, Y1], 
        #          "b", linestyle="--", linewidth=0.5)
        plt.pause(0.001)