import jax
import jax.numpy as jnp
from tqdm import tqdm
import math
import os
import json
import iblbm
import matplotlib.pyplot as plt
import matplotlib as mpl

working_dir = os.path.dirname(os.path.abspath(__file__))

# read the parameters from json file
with open(working_dir + "/parameters.json") as f:
    params = json.load(f)

D = params["D"]
U = params["U"]
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

# define LBM parameters
TAU = 3 * NU + 0.5  # relaxation time
OMEGA = 1 / TAU  # relaxation parameter
MRT_OMEGA = iblbm.get_omega_mrt(OMEGA)
ARC_LEN = D * math.pi / N_MARKER  # arc length between the markers
X1 = int(X_OBJ - 1.5 * D)
X2 = int(X_OBJ + 1.5 * D)
Y1 = int(Y_OBJ - 1.5 * D)
Y2 = int(Y_OBJ + 1.5 * D)

# ----------------- initialize properties -----------------

# macroscopic properties
rho = jnp.ones((NX, NY), dtype=jnp.float32)
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)
u = u.at[0].set(U)

# microscopic properties
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)
f = iblbm.equilibrum(rho, u, f)
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)

# structural dynamics
d = jnp.zeros((2), dtype=jnp.float32)
v = jnp.zeros((2), dtype=jnp.float32)
a = jnp.zeros((2), dtype=jnp.float32)

# mesh grid
X, Y = jnp.meshgrid(
    jnp.arange(NX, dtype=jnp.int16), jnp.arange(NY, dtype=jnp.int16), indexing="ij"
)
X_MARKER = X_OBJ + 0.5 * D * jnp.cos(
    jnp.linspace(0, jnp.pi * 2, N_MARKER, endpoint=False)
)
Y_MARKER = Y_OBJ + 0.5 * D * jnp.sin(
    jnp.linspace(0, jnp.pi * 2, N_MARKER, endpoint=False)
)

# main loop
@jax.jit
def update(f, feq, rho, u, d, v, a):
    """Update distribution functions for one time step"""

    # new macroscopic (uncorrected)
    rho, u = iblbm.get_macroscopic(f, rho, u)

    # Compute correction forces (Immersed Boundary Method)
    g_markers = jnp.zeros((2, N_MARKER))
    g_fluid = jnp.zeros((2, X2 - X1, Y2 - Y1))

    for _ in range(1):
        g_markers_diff = jnp.zeros((2, N_MARKER))  # force to the marker
        g_fluid_diff = jnp.zeros((2, X2 - X1, Y2 - Y1))  # force to the fluid

        for i in range(N_MARKER):
            x_marker = X_MARKER[i] + d[0]  # x coordinate of the marker
            y_marker = Y_MARKER[i] + d[1]  # y coordinate of the marker            
            kernel = iblbm.kernel3(x_marker, y_marker, X[X1:X2, Y1:Y2], Y[X1:X2, Y1:Y2])

            # velocity interpolation (at markers)
            u_marker = iblbm.intepolate(u[:, X1:X2, Y1:Y2], kernel)

            # compute correction force (at markers)
            g_marker_diff = iblbm.get_correction_g(v, u_marker)

            # accumulate correction forces
            g_markers_diff = g_markers_diff.at[:, i].set(- g_marker_diff)            
            g_fluid_diff += iblbm.spreading(g_marker_diff, kernel)

        # velocity correction
        u = u.at[:, X1:X2, Y1:Y2].add(iblbm.get_correction_u(g_fluid_diff))

        # record the total correction force to the fluid and markers
        g_fluid += g_fluid_diff
        g_markers += g_markers_diff

    # Compute dynamics
    g_obj = jnp.sum(g_markers, axis=1) * ARC_LEN
    
    a, v, d = iblbm.newmark(a, v, d, g_obj, M, K, C)

    # Compute equilibrium
    feq = iblbm.equilibrum(rho, u, feq)

    # Collision with forces
    f = iblbm.collision_mrt(f, feq, MRT_OMEGA)
    f = f.at[:, X1:X2, Y1:Y2].add(iblbm.get_correction_f(u[:, X1:X2, Y1:Y2], g_fluid, OMEGA,))

    # Streaming
    f = iblbm.streaming(f)

    # Outlet BC at right wall (No gradient BC)
    f = iblbm.right_outlet(f)

    # Inlet BC at left wall (Non-equilibrium Bounce-Back)
    f, rho, u = iblbm.left_inlet(f, rho, u, U)

    return f, feq, rho, u, d, v, a, g_obj

PLOT = True
PLOT_EVERY = 100
PLOT_AFTER = 500
N_PLOTS = int((TM - PLOT_AFTER) // PLOT_EVERY)
PLOT_CONTENT = "curl"

if PLOT:
    plt.figure(figsize=(8, 4))
    
for t in tqdm(range(TM)):
    f, feq, rho, u, d, v, a, g = update(f, feq, rho, u, d, v, a)

    
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
        plt.clf()

        if PLOT_CONTENT == "curl":

            @jax.jit
            def calculate_curl(u):
                ux_y = jnp.gradient(u[0], axis=1)
                uy_x = jnp.gradient(u[1], axis=0)
                curl = ux_y - uy_x
                return curl

            curl = calculate_curl(u)
            plt.imshow(
                curl.T,
                cmap="seismic",
                # cmap="jet",
                aspect="equal",
                norm=mpl.colors.CenteredNorm(),
            )
        
        if PLOT_CONTENT == "rho":
            plt.imshow(
                rho.T,
                cmap="seismic",
                aspect="equal",
                norm=mpl.colors.CenteredNorm(vcenter=1),
            )
            
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
        # plt.title(fr"$t\times f_n$= {t*FN:.0f}")
        plt.axvline(X_OBJ, color="k", linestyle="--", linewidth=0.5)
        plt.axhline(Y_OBJ, color="k", linestyle="--", linewidth=0.5)
        
        # draw outline of the IBM region as a rectangle
        plt.plot([X1, X1, X2, X2, X1], 
                 [Y1, Y2, Y2, Y1, Y1], 
                 "b", linestyle="--", linewidth=0.5)
        plt.pause(0.001)