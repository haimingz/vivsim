# ! this script is not working properly yet !

import math
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from vivsim import dyn, ib, lbm, post

import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    # '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)


# physics parameters for viv
RE = 500  # Reynolds number
UR = 5  # Reduced velocity
MR = 10  # Mass ratio
ZETA = 0   # Damping ratio

# geometric parameters
D = 20   # Cylinder diameter
NX = 40 * D  # Number of grid points in x direction
NY = 30 * D   # Number of grid points in y direction
X_OBJ = 20 * D   # x-coordinate of the cylinder
Y_OBJ = 15 * D   # y-coordinate of the cylinder
N_MARKER_EACH = 4 * D   # Number of markers on the circle

# time parameters
U0 = 0.05  # Inlet velocity
TM = 60000   # Maximum number of time steps

# plot options
PLOT = True  # whether to plot the results
PLOT_EVERY = 100  # plot every n time steps
PLOT_AFTER = 00  # plot after n time steps

# derived parameters 
NU = U0 * D / RE  # kinematic viscosity
FN = U0 / (UR * D)  # natural frequency
M = math.pi * (D / 2) ** 2 * MR  # mass of the cylinder
K = (FN * 2 * math.pi) ** 2 * M * (1 + 1 / MR)  # stiffness
C = 2 * math.sqrt(K * M) * ZETA  # damping

# parameters for IB-LBM
TAU = 3 * NU + 0.5  # relaxation time
OMEGA = 1 / TAU  # relaxation parameter
MRT_OMEGA = lbm.get_omega_mrt(OMEGA)   # relaxation matrix for MRT
L_ARC = D * math.pi / N_MARKER_EACH  # arc length between the markers
RE_GRID = RE / D  # Reynolds number based on grid size
X1 = int(X_OBJ - 5 * D)   # left boundary of the IBM region 
X2 = int(X_OBJ + 7 * D)   # right boundary of the IBM region
Y1 = int(Y_OBJ - 6 * D)   # bottom boundary of the IBM region
Y2 = int(Y_OBJ + 6 * D)   # top boundary of the IBM region
MDF = 1  # number of iterations for multi-direct forcing

# generate mesh grid
X, Y = jnp.meshgrid(jnp.arange(NX, dtype=jnp.uint16), 
                    jnp.arange(NY, dtype=jnp.uint16), 
                    indexing="ij")

THETA_MAKERS = jnp.linspace(0, jnp.pi * 2, N_MARKER_EACH, dtype=jnp.float32, endpoint=False)

SPACING = 4 * D # spacing between the cylinders and the center point

angle = jnp.pi / 8
X_MARKERS_1 = X_OBJ + SPACING * jnp.cos(angle) + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS_1 = Y_OBJ + SPACING * jnp.sin(angle) + 0.5 * D * jnp.sin(THETA_MAKERS)

angle += jnp.pi / 2
X_MARKERS_2 = X_OBJ + SPACING * jnp.cos(angle) + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS_2 = Y_OBJ + SPACING * jnp.sin(angle) + 0.5 * D * jnp.sin(THETA_MAKERS)

angle += jnp.pi / 2
X_MARKERS_3 = X_OBJ + SPACING * jnp.cos(angle) + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS_3 = Y_OBJ + SPACING * jnp.sin(angle) + 0.5 * D * jnp.sin(THETA_MAKERS)

angle += jnp.pi / 2
X_MARKERS_4 = X_OBJ + SPACING * jnp.cos(angle) + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS_4 = Y_OBJ + SPACING * jnp.sin(angle) + 0.5 * D * jnp.sin(THETA_MAKERS)

X_MARKERS = jnp.concatenate([X_MARKERS_1, X_MARKERS_2, X_MARKERS_3, X_MARKERS_4])
Y_MARKERS = jnp.concatenate([Y_MARKERS_1, Y_MARKERS_2, Y_MARKERS_3, Y_MARKERS_4])
N_MARKER = 4 * N_MARKER_EACH


# plt.plot(X_MARKERS, Y_MARKERS, ".")
# plt.plot(X_OBJ, Y_OBJ, "x")
# plt.plot(jnp.array([X1, X1, X2, X2, X1]), 
#              jnp.array([Y1, Y2, Y2, Y1, Y1]), 
#              "b", linestyle="--", linewidth=0.5)
# plt.xlim(0, NX)
# plt.ylim(0, NY)
# plt.gca().set_aspect("equal")
# plt.show()

INERTIA = 4 * (M / 4) * SPACING ** 2  # moment of inertia

K_ROT = 500

M_MATRIX = jnp.diag(jnp.array([M, M, INERTIA]))
K_MATRIX = jnp.diag(jnp.array([K, K, K_ROT]))
C_MATRIX = jnp.diag(jnp.array([C, C, 500]))

# create empty arrays
rho = jnp.ones((NX, NY), dtype=jnp.float32)  # density of fluid
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)  # velocity of fluid
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # distribution functions
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # equilibrium distribution functions
d = jnp.zeros((3), dtype=jnp.float32)  # displacement
v = jnp.zeros((3), dtype=jnp.float32)  # velocity
a = jnp.zeros((3), dtype=jnp.float32)  # acceleration
h = jnp.zeros((3), dtype=jnp.float32)  # force

# initialize
u = u.at[0].set(U0)
f = lbm.get_equilibrum(rho, u, f)
rho_next = jnp.ones((NY), dtype=jnp.float32)
u_next = jnp.zeros((2, NY), dtype=jnp.float32)

# define main loop 
@jax.jit
def update(f, feq, rho, u, d, v, a, h, rho_right_next, u_right_next):

    # Immersed Boundary Method
    x_markers, y_markers = ib.update_markers_coords_3dof(X_MARKERS, Y_MARKERS, X_OBJ, Y_OBJ, d)
    v_markers = ib.update_markers_velocity_3dof(x_markers, y_markers, X_OBJ, Y_OBJ, d, v)
    h_markers, g, u = ib.multi_direct_forcing(u, X, Y, v_markers, x_markers, y_markers, X1, X2, Y1, Y2, MDF)

    # Compute force to the obj (including internal fluid force)
    h = h.at[:2].set(ib.calculate_force_obj(h_markers, L_ARC))
    h = h.at[2].set(ib.calculate_torque_obj(x_markers, y_markers, X_OBJ, Y_OBJ, d, h_markers, L_ARC))
    
    # eliminate internal fluid force (Feng's rigid body approximation)
    # h = h.at[:2].add(a * math.pi * D ** 2 / 4)  # found unstable for high Re
    
    # Compute solid dynamics
    a, v, d = dyn.newmark3dof(a, v, d, h, M_MATRIX, K_MATRIX, C_MATRIX)

    # Compute equilibrium
    feq = lbm.get_equilibrum(rho, u, feq)

    # Collision
    f = lbm.collision_mrt(f, feq, MRT_OMEGA)
    
    # Add source term
    f = f.at[:, X1:X2, Y1:Y2].add(ib.get_source(u[:, X1:X2, Y1:Y2], g, OMEGA))

    # non-reflecting bounday condition 
   
    # u = u.at[:,-1].set(u_right_next) 
    # rho = rho.at[-1].set(rho_right_next)    
    f = f.at[:,-1].set(lbm.get_equilibrum(rho_right_next, u_right_next, feq[:,-1]))
    
    rhot_right, ut_right = lbm.cbc_right(rho, u)
    rho_right_next = rho[-1] + rhot_right
    u_right_next = u[:,-1] + ut_right
    
    # Streaming
    f = lbm.streaming(f)

    # Set Outlet BC at right wall (No gradient BC)
    # f = lbm.right_outlet(f)

    # Set Inlet BC at left wall (Zou/He scheme)
    f, rho = lbm.left_velocity(f, rho, U0, 0)

    # update new macroscopic
    rho, u = lbm.get_macroscopic(f, rho, u)
     
    return f, feq, rho, u, d, v, a, h, rho_right_next, u_right_next

# create the plot template
if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    
    plt.figure(figsize=(8, 4))
    
    curl = post.calculate_curl(u)
    im = plt.imshow(
        curl.T,
        # rho.T,
        extent=[0, NX/D, 0, NY/D],
        cmap="seismic",
        aspect="equal",
        origin="lower",
        # norm=mpl.colors.CenteredNorm(),
        vmax=0.05,
        vmin=-0.05,
    )

    plt.colorbar()
    
    # plt.xticks([])
    # plt.yticks([])
    plt.xlabel("x/D")
    plt.ylabel("y/D")

    # draw athe geometry
    x_center1 = X_OBJ + SPACING * jnp.cos(jnp.pi / 8 + jnp.pi / 2 * 0 + d[2]) + d[0]
    y_center1 = Y_OBJ + SPACING * jnp.sin(jnp.pi / 8 + jnp.pi / 2 * 0 + d[2]) + d[1]
    x_center2 = X_OBJ + SPACING * jnp.cos(jnp.pi / 8 + jnp.pi / 2 * 1 + d[2]) + d[0]
    y_center2 = Y_OBJ + SPACING * jnp.sin(jnp.pi / 8 + jnp.pi / 2 * 1 + d[2]) + d[1]
    x_center3 = X_OBJ + SPACING * jnp.cos(jnp.pi / 8 + jnp.pi / 2 * 2 + d[2]) + d[0]
    y_center3 = Y_OBJ + SPACING * jnp.sin(jnp.pi / 8 + jnp.pi / 2 * 2 + d[2]) + d[1]
    x_center4 = X_OBJ + SPACING * jnp.cos(jnp.pi / 8 + jnp.pi / 2 * 3 + d[2]) + d[0]
    y_center4 = Y_OBJ + SPACING * jnp.sin(jnp.pi / 8 + jnp.pi / 2 * 3 + d[2]) + d[1]    
    
    circle1 = plt.Circle((x_center1 / D, y_center1 / D), 0.5, 
                        edgecolor='black', linewidth=0.5,
                        facecolor='white', fill=True)
    circle2 = plt.Circle((x_center2 / D, y_center2 / D), 0.5, 
                        edgecolor='black', linewidth=0.5,
                        facecolor='white', fill=True)
    circle3 = plt.Circle((x_center3 / D, y_center3 / D), 0.5, 
                        edgecolor='black', linewidth=0.5,
                        facecolor='white', fill=True)
    circle4 = plt.Circle((x_center4 / D, y_center4 / D), 0.5, 
                        edgecolor='black', linewidth=0.5,
                        facecolor='white', fill=True)
    plt.gca().add_artist(circle1)
    plt.gca().add_artist(circle2)
    plt.gca().add_artist(circle3)
    plt.gca().add_artist(circle4)
    
    line1 = plt.Line2D([x_center1 / D, x_center2 / D], [y_center1 / D, y_center2 / D], color="black", linewidth=0.5)
    line2 = plt.Line2D([x_center2 / D, x_center3 / D], [y_center2 / D, y_center3 / D], color="black", linewidth=0.5)
    line3 = plt.Line2D([x_center3 / D, x_center4 / D], [y_center3 / D, y_center4 / D], color="black", linewidth=0.5)
    line4 = plt.Line2D([x_center4 / D, x_center1 / D], [y_center4 / D, y_center1 / D], color="black", linewidth=0.5)
    
    plt.gca().add_artist(line1)
    plt.gca().add_artist(line2)
    plt.gca().add_artist(line3)
    plt.gca().add_artist(line4)
                
    # draw the central lines
    plt.axvline(X_OBJ / D, color="k", linestyle="--", linewidth=0.5)
    plt.axhline(Y_OBJ / D, color="k", linestyle="--", linewidth=0.5)
    
    # draw outline of the IBM region as a rectangle
    # plt.plot(jnp.array([X1, X1, X2, X2, X1]) / D, 
    #          jnp.array([Y1, Y2, Y2, Y1, Y1]) / D, 
    #          "b", linestyle="--", linewidth=0.5)
    
    plt.tight_layout()

# start simulation 
for t in tqdm(range(TM)):
    f, feq, rho, u, d, v, a, h, rho_next, u_next = update(f, feq, rho, u, d, v, a, h, rho_next, u_next)
    
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:

        im.set_data(post.calculate_curl(u).T)
        # im.set_data(rho.T)
        # im.autoscale()
        
        x_center1 = X_OBJ + SPACING * jnp.cos(jnp.pi / 8 + jnp.pi / 2 * 0 + d[2]) + d[0]
        y_center1 = Y_OBJ + SPACING * jnp.sin(jnp.pi / 8 + jnp.pi / 2 * 0 + d[2]) + d[1]
        x_center2 = X_OBJ + SPACING * jnp.cos(jnp.pi / 8 + jnp.pi / 2 * 1 + d[2]) + d[0]
        y_center2 = Y_OBJ + SPACING * jnp.sin(jnp.pi / 8 + jnp.pi / 2 * 1 + d[2]) + d[1]
        x_center3 = X_OBJ + SPACING * jnp.cos(jnp.pi / 8 + jnp.pi / 2 * 2 + d[2]) + d[0]
        y_center3 = Y_OBJ + SPACING * jnp.sin(jnp.pi / 8 + jnp.pi / 2 * 2 + d[2]) + d[1]
        x_center4 = X_OBJ + SPACING * jnp.cos(jnp.pi / 8 + jnp.pi / 2 * 3 + d[2]) + d[0]
        y_center4 = Y_OBJ + SPACING * jnp.sin(jnp.pi / 8 + jnp.pi / 2 * 3 + d[2]) + d[1]
        
        circle1.center = (x_center1 / D, y_center1 / D)
        circle2.center = (x_center2 / D, y_center2 / D)
        circle3.center = (x_center3 / D, y_center3 / D)
        circle4.center = (x_center4 / D, y_center4 / D)
        
        line1.set_xdata([x_center1 / D, x_center2 / D])
        line1.set_ydata([y_center1 / D, y_center2 / D])
        line2.set_xdata([x_center2 / D, x_center3 / D])
        line2.set_ydata([y_center2 / D, y_center3 / D])
        line3.set_xdata([x_center3 / D, x_center4 / D])
        line3.set_ydata([y_center3 / D, y_center4 / D])
        line4.set_xdata([x_center4 / D, x_center1 / D])
        line4.set_ydata([y_center4 / D, y_center1 / D])
        
        plt.pause(0.001)

