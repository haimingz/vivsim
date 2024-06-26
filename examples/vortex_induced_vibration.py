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
from vivsim import dyn, ib, lbm, post


# physics parameters for viv
RE = 1000  # Reynolds number
UR = 5  # Reduced velocity
MR = 10  # Mass ratio
ZETA = 0   # Damping ratio
D = 50   # Cylinder diameter
U0 = 0.05  # Inlet velocity

# domain settings
NX = 800  # Number of grid points in x direction
NY = 400   # Number of grid points in y direction
X_OBJ = 300   # x-coordinate of the cylinder
Y_OBJ = 200   # y-coordinate of the cylinder
N_MARKER = 100   # Number of markers on the circle
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
L_ARC = D * math.pi / N_MARKER  # arc length between the markers
RE_GRID = RE / D  # Reynolds number based on grid size
X1 = int(X_OBJ - 0.7 * D)   # left boundary of the IBM region
X2 = int(X_OBJ + 1.0 * D)   # right boundary of the IBM region
Y1 = int(Y_OBJ - 1.5 * D)   # bottom boundary of the IBM region
Y2 = int(Y_OBJ + 1.5 * D)   # top boundary of the IBM region
MDF = 3  # number of iterations for multi-direct forcing

# generate mesh grid
X, Y = jnp.meshgrid(jnp.arange(NX, dtype=jnp.int16), 
                    jnp.arange(NY, dtype=jnp.int16), 
                    indexing="ij")

THETA_MAKERS = jnp.linspace(0, jnp.pi * 2, N_MARKER, endpoint=False)
X_MARKERS = X_OBJ + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS = Y_OBJ + 0.5 * D * jnp.sin(THETA_MAKERS)

# create empty arrays
rho = jnp.ones((NX, NY), dtype=jnp.float32)  # density of fluid
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)  # velocity of fluid
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # distribution functions
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # equilibrium distribution functions
d = jnp.zeros((2), dtype=jnp.float32)  # displacement
v = jnp.zeros((2), dtype=jnp.float32)  # velocity
a = jnp.zeros((2), dtype=jnp.float32)  # acceleration
g = jnp.zeros((2), dtype=jnp.float32)  # force

# initialize
u = u.at[0].set(U0)
f = lbm.get_equilibrum(rho, u, f)

# define main loop 
@jax.jit
def update(f, feq, rho, u, d, v, a, g):

    # Immersed Boundary Method
    g_to_markers = jnp.zeros((N_MARKER, 2))  # force to the markers
    g_to_fluid = jnp.zeros((2, X2 - X1, Y2 - Y1))  # force to the fluid
    
    # calculate the kernels
    x_markers = X_MARKERS + d[0]  # x coordinates of the markers
    y_markers = Y_MARKERS + d[1]  # y coordinates of the markers
    kernels = jax.vmap(ib.kernel3, in_axes=(0, 0, None, None))(x_markers, y_markers, X[X1:X2, Y1:Y2], Y[X1:X2, Y1:Y2])
        
    for _ in range(MDF):
        
        # velocity interpolation (at markers)
        u_markers = jax.vmap(ib.interpolate_u, in_axes=(None, 0))(u[:, X1:X2, Y1:Y2], kernels)
        
        # compute correction force (at markers) 
        g_needed = jax.vmap(ib.get_g_correction, in_axes=(None, 0))(v, u_markers)
        g_needed_spread = jnp.sum(jax.vmap(ib.spread_g, in_axes=(0, 0))(g_needed, kernels), axis=0)
        
        # velocity correction
        u = u.at[:, X1:X2, Y1:Y2].add(ib.get_u_correction(g_needed_spread))
        
        # accumulate the coresponding correction force to the markers and the fluid
        g_to_markers += - g_needed
        g_to_fluid += g_needed_spread

    # Compute force to the obj (including internal fluid force)
    g = jnp.sum(g_to_markers, axis=0) * L_ARC
    
    # eliminate internal fluid force (Feng's rigid body approximation)
    g += a * math.pi * D ** 2 / 4  # found unstable for high Re
    
    # Compute solid dynamics
    a, v, d = dyn.newmark(a, v, d, g, M, K, C)

    # Compute equilibrium
    feq = lbm.get_equilibrum(rho, u, feq)

    # Collision
    f = lbm.collision_mrt(f, feq, MRT_OMEGA)
    
    # Add source term
    f = f.at[:, X1:X2, Y1:Y2].add(ib.get_source(u[:, X1:X2, Y1:Y2], g_to_fluid, OMEGA))

    # Streaming
    f = lbm.streaming(f)

    # Set Outlet BC at right wall (No gradient BC)
    f = lbm.right_outlet(f)

    # Set Inlet BC at left wall (Zou/He scheme)
    f, rho = lbm.left_velocity(f, rho, U0, 0)

    # update new macroscopic
    rho, u = lbm.get_macroscopic(f, rho, u)
     
    return f, feq, rho, u, d, v, a, g

# create the plot template
if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    
    plt.figure(figsize=(8, 4))
    
    curl = post.calculate_curl(u)
    im = plt.imshow(
        curl.T,
        extent=[0, NX/D, 0, NY/D],
        cmap="seismic",
        aspect="equal",
        origin="lower",
        norm=mpl.colors.CenteredNorm(),
        # vmax=0.03,
        # vmin=-0.03,
    )

    plt.colorbar()
    
    # plt.xticks([])
    # plt.yticks([])
    plt.xlabel("x/D")
    plt.ylabel("y/D")

    # draw a circle representing the cylinder
    circle = plt.Circle(((X_OBJ + d[0]) / D, (Y_OBJ + d[1]) / D), 0.5, 
                        edgecolor='black', linewidth=0.5,
                        facecolor='white', fill=True)
    plt.gca().add_artist(circle)
                
    # draw the central lines
    plt.axvline(X_OBJ / D, color="k", linestyle="--", linewidth=0.5)
    plt.axhline(Y_OBJ / D, color="k", linestyle="--", linewidth=0.5)
    
    # draw outline of the IBM region as a rectangle
    # plt.plot([X1, X1, X2, X2, X1], 
    #          [Y1, Y2, Y2, Y1, Y1], 
    #          "b", linestyle="--", linewidth=0.5)
    
    plt.tight_layout()

# start simulation 
for t in tqdm(range(TM)):
    f, feq, rho, u, d, v, a, g = update(f, feq, rho, u, d, v, a, g)
    
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:

        im.set_data(post.calculate_curl(u).T)
        im.autoscale()
        circle.center = ((X_OBJ + d[0]) / D, (Y_OBJ + d[1]) / D)
        
        plt.pause(0.001)

