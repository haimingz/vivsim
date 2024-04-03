import jax
import jax.numpy as jnp
from tqdm import tqdm
import math
import os
import json
import iblbm, dynamics
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
NT = params["TM"]
N_MARKER = params["N_MARKER"]
X_OBJ = params["X_CYLINDER"]
Y_OBJ = params["Y_CYLINDER"]


# coordinates
X_MESH, Y_MESH = jnp.meshgrid(
    jnp.arange(NX, dtype=jnp.int16), jnp.arange(NY, dtype=jnp.int16), indexing="ij"
)
X_MARKER = X_OBJ + 0.5 * D * jnp.cos(
    jnp.linspace(0, jnp.pi * 2, N_MARKER, endpoint=False)
)
Y_MARKER = Y_OBJ + 0.5 * D * jnp.sin(
    jnp.linspace(0, jnp.pi * 2, N_MARKER, endpoint=False)
)

# ------------------ Lattice Boltzmann Method (LBM) ------------------

TAU = 3 * NU + 0.5  # relaxation time
OMEGA = 1 / TAU  # relaxation parameter
ARC_LEN = D * math.pi / N_MARKER  # arc length between the markers
# IBM region
IBM_X1 = int(X_OBJ - 1.0 * D)
IBM_X2 = int(X_OBJ + 1.5 * D)
IBM_Y1 = int(Y_OBJ - 1.5 * D)
IBM_Y2 = int(Y_OBJ + 1.5 * D)
MRT_OMEGA = iblbm.generate_omega_mrt(OMEGA)

# ----------------- initialize properties -----------------

# microscopic properties
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)

# macroscopic properties
rho = jnp.ones((NX, NY), dtype=jnp.float32)
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)
u = u.at[0].set(U)

f = iblbm.equilibrum(rho, u, f)

# structural dynamics
d = jnp.zeros((2), dtype=jnp.float32)
v = jnp.zeros((2), dtype=jnp.float32)
a = jnp.zeros((2), dtype=jnp.float32)


# main loop
@jax.jit
def update(f, feq, rho, u, d, v, a):
    """Update distribution functions for one time step"""

    # new macroscopic (uncorrected)
    rho, u = iblbm.get_macroscopic(f, rho, u)

    # Compute correction forces (Immersed Boundary Method)
    g_fluid_sum = jnp.zeros((2, NX, NY))
    g_markers_sum = jnp.zeros((2, N_MARKER))

    for _ in range(3):
        g_markers = jnp.zeros((2, N_MARKER))  # force to the marker
        g_fluid = jnp.zeros((2, NX, NY))  # force to the fluid

        for i in range(N_MARKER):
            x_marker = X_MARKER[i] + d[0]  # x coordinate of the marker
            y_marker = Y_MARKER[i] + d[1]  # y coordinate of the marker
            kernel = iblbm.kernel3(X_MESH[IBM_X1:IBM_X2, IBM_Y1:IBM_Y2] - x_marker) \
                * iblbm.kernel3(Y_MESH[IBM_X1:IBM_X2, IBM_Y1:IBM_Y2] - y_marker)

            # velocity interpolation (at markers)
            u_at_marker = jnp.einsum("xy,dxy->d", kernel, u[:, IBM_X1:IBM_X2, IBM_Y1:IBM_Y2])  # shape (2)

            # compute correction force (at markers)
            g_needed_at_marker = 2 * (v - u_at_marker)  # shape (2)

            # accumulate correction forces
            g_markers = g_markers.at[:, i].set(-g_needed_at_marker)  # force at markers
            g_fluid = g_fluid.at[0, IBM_X1:IBM_X2, IBM_Y1:IBM_Y2].add(
                kernel * g_needed_at_marker[0]
            )  # spread force to fluid
            g_fluid = g_fluid.at[1, IBM_X1:IBM_X2, IBM_Y1:IBM_Y2].add(
                kernel * g_needed_at_marker[1]
            )  # spread force to fluid

        # velocity correction
        u = u.at[:, IBM_X1:IBM_X2, IBM_Y1:IBM_Y2].add(g_fluid[:, IBM_X1:IBM_X2, IBM_Y1:IBM_Y2] * 0.5)

        # record the total correction force to the fluid and markers
        g_fluid_sum += g_fluid
        g_markers_sum += g_markers

    # Compute dynamics
    g_obj = jnp.sum(g_markers_sum, axis=1) * ARC_LEN
    
    ax, vx, dx = dynamics.newmark(a[0], v[0], d[0], g_obj[0], M, K, C)
    # ax, vx, dx = 0, 0, 0
    ay, vy, dy = dynamics.newmark(a[1], v[1], d[1], g_obj[1], M, K, C)
    a = jnp.array([ax, ay])
    v = jnp.array([vx, vy])
    d = jnp.array([dx, dy])

    # Compute discretized correction force
    h = jnp.zeros((9, NX, NY))
    h = iblbm.discretize_force(u, g_fluid_sum, h)

    # Compute equilibrium
    feq = iblbm.equilibrum(rho, u, feq)

    # Collision with forces
    f = iblbm.collision_mrt(f, feq, MRT_OMEGA)
    f = iblbm.post_collision_correction(f, h, OMEGA)

    # Streaming
    f = iblbm.streaming(f)

    # Outlet BC at right wall (No gradient BC)
    f = iblbm.right_outlet(f)

    # Inlet BC at left wall (Non-equilibrium Bounce-Back)
    f, rho, u = iblbm.left_inlet(f, rho, u, U)

    return f, feq, rho, u, d, v, a, g_obj

PLOT = True
PLOT_EVERY = 100
PLOT_AFTER = 5000
N_PLOTS = int((NT - PLOT_AFTER) // PLOT_EVERY)
PLOT_CONTENT = "curl"

if PLOT:
    plt.figure(figsize=(8, 4))
    
for t in tqdm(range(NT)):
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
        plt.plot([IBM_X1, IBM_X1, IBM_X2, IBM_X2, IBM_X1], 
                 [IBM_Y1, IBM_Y2, IBM_Y2, IBM_Y1, IBM_Y1], 
                 "b", linestyle="--", linewidth=0.5)
        plt.pause(0.001)