import os
import math
import yaml

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

from vivsim import dyn, ib, lbm, post


ROOT_PATH = os.path.dirname(__file__) + "/case_demo/"  # path to save the output 

# ----------------------- load parameters from config file -----------------------

with open(ROOT_PATH + "parameters.yml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

RE = params["RE"]  # Reynolds number *
UR = params["UR"]  # reduced velocity *
MR = params["MR"]   # mass ratio *
ZETA = params["ZETA"]  # damping ratio *
D = params["D"]  # diameter *
U0 = params["U0"]  # inlet velocity *
NX = params["NX"]  # number of grid points in x direction *
NY = params["NY"]  # number of grid points in y direction *
X_OBJ = params["X_OBJ"]  # y coordinate of the cylinder *
Y_OBJ = params["Y_OBJ"]  # x coordinate of the cylinder *
N_MARKER = params["N_MARKER"]  # number of Lagrangian markers *
TM = params["TM"]  # number of time steps *

SAVE = params["SAVE"]  # whether to save the results
SAVE_EVERY = params["SAVE_EVERY"]  # save every n time steps
SAVE_AFTER = params["SAVE_AFTER"] # save after n time steps
SAVE_DSAMP = params["SAVE_DSAMP"] # spatially downsample the output by n
if params["SAVE_FORMAT"] == "f32":
    SAVE_FORMAT = jnp.float32  # precision of the output
else:
    SAVE_FORMAT = jnp.float16  # precision of the output

PLOT = params["PLOT"]  # whether to plot the results
PLOT_EVERY = params["PLOT_EVERY"]  # plot every n time steps
PLOT_AFTER = params["PLOT_AFTER"]  # plot after n time steps
PLOT_CONTENT = params["PLOT_CONTENT"]  # plot the curl or density

# ----------------------- derived dimensional parameters -----------------------

NU = U0 * D / RE  # kinematic viscosity
FN = U0 / (UR * D)  # natural frequency
M = math.pi * (D / 2) ** 2 * MR  # mass of the cylinder
K = (FN * 2 * math.pi) ** 2 * M * (1 + 1 / MR)  # stiffness
C = 2 * math.sqrt(K * M) * ZETA  # damping

# ----------------------- determine IB-LBM parameters -----------------------

TAU = 3 * NU + 0.5  # relaxation time
OMEGA = 1 / TAU  # relaxation parameter
MRT_OMEGA = lbm.get_omega_mrt(OMEGA)   # relaxation matrix for MRT
L_ARC = D * math.pi / N_MARKER  # arc length between the markers
RE_GRID = RE / D  # Reynolds number based on grid size
X1 = int(X_OBJ - 0.7 * D)   # left boundary of the IBM region
X2 = int(X_OBJ + 1.0 * D)   # right boundary of the IBM region
Y1 = int(Y_OBJ - 1.5 * D)   # bottom boundary of the IBM region
Y2 = int(Y_OBJ + 1.5 * D)   # top boundary of the IBM region
MDF_ITER = 1  # number of iterations for multi-direct forcing

# ----------------------- initialize -----------------------

# generate mesh grid
X, Y = jnp.meshgrid(jnp.arange(NX, dtype=jnp.int16), 
                    jnp.arange(NY, dtype=jnp.int16), 
                    indexing="ij")

THETA_MAKERS = jnp.linspace(0, jnp.pi * 2, N_MARKER, endpoint=False)
X_MARKERS = X_OBJ + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS = Y_OBJ + 0.5 * D * jnp.sin(THETA_MAKERS)

# macroscopic properties
rho = jnp.ones((NX, NY), dtype=jnp.float32)  # density of fluid
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)  # velocity of fluid
u = u.at[0].set(U0)

# microscopic properties
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # distribution functions
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # equilibrium distribution functions
f = lbm.get_equilibrum(rho, u, f)

# structural dynamics properties
d = jnp.zeros((2), dtype=jnp.float32)  # displacement
v = jnp.zeros((2), dtype=jnp.float32)  # velocity
a = jnp.zeros((2), dtype=jnp.float32)  # acceleration
g = jnp.zeros((2), dtype=jnp.float32)  # force

# ----------------------- define main loop -----------------------
@jax.jit
def update(f, feq, rho, u, d, v, a, g):

    # Immersed Boundary Method
    g_to_markers = jnp.zeros((N_MARKER, 2))  # force to the markers
    g_to_fluid = jnp.zeros((2, X2 - X1, Y2 - Y1))  # force to the fluid
    
    # calculate the kernels
    x_markers = X_MARKERS + d[0]  # x coordinates of the markers
    y_markers = Y_MARKERS + d[1]  # y coordinates of the markers
    kernels = jax.vmap(ib.kernel3, in_axes=(0, 0, None, None))(x_markers, y_markers, X[X1:X2, Y1:Y2], Y[X1:X2, Y1:Y2])
        
    for _ in range(MDF_ITER):
        
        # interpolate velocity at the markers
        u_markers = jax.vmap(ib.interpolate_u, in_axes=(None, 0))(u[:, X1:X2, Y1:Y2], kernels)
        
        # calculate and apply the needed correction force to the fluid
        g_needed = jax.vmap(ib.get_g_correction, in_axes=(None, 0))(v, u_markers)
        g_needed_spread = jnp.sum(jax.vmap(ib.spread_g, in_axes=(0, 0))(g_needed, kernels), axis=0)
        u = u.at[:, X1:X2, Y1:Y2].add(ib.get_u_correction(g_needed_spread))
        
        # accumulate the coresponding correction force to the markers and the fluid
        g_to_markers += - g_needed
        g_to_fluid += g_needed_spread

    # Compute force to the obj (including internal fluid force)
    g = jnp.sum(g_to_markers, axis=0) * L_ARC
    
    # eliminate internal fluid force (Feng’s rigid body approximation)
    # g += a * math.pi * D ** 2 / 4  # found unstable for high Re
    
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

# ----------------------- template for ploting -----------------------

if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    
    plt.figure(figsize=(8, 4))
    
    if PLOT_CONTENT == "curl":
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
        
    if PLOT_CONTENT == "rho":
        im = plt.imshow(
            rho.T,
            extent=[0, NX/D, 0, NY/D],
            cmap="seismic",
            aspect="equal",
            origin="lower",
            # norm=mpl.colors.CenteredNorm(vcenter=1),
            # vmax=1.05,
            # vmin=0.95,
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

if SAVE:
    N_SAVE = int((TM - SAVE_AFTER) / SAVE_EVERY)
    d_save = jnp.zeros((N_SAVE, 2), dtype=SAVE_FORMAT)
    v_save = jnp.zeros((N_SAVE, 2), dtype=SAVE_FORMAT)
    a_save = jnp.zeros((N_SAVE, 2), dtype=SAVE_FORMAT)
    g_save = jnp.zeros((N_SAVE, 2), dtype=SAVE_FORMAT)
    save_pointer = 0

# ----------------------- start simulation -----------------------

for t in tqdm(range(TM)):
    f, feq, rho, u, d, v, a, g = update(f, feq, rho, u, d, v, a, g)
    
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:

        if PLOT_CONTENT == "curl":
            im.set_data(post.calculate_curl(u).T)
            im.autoscale()
            circle.center = ((X_OBJ + d[0]) / D, (Y_OBJ + d[1]) / D)
        
        if PLOT_CONTENT == "rho":
            im.set_data(rho.T)
            im.autoscale()
            circle.center = ((X_OBJ + d[0]) / D, (Y_OBJ + d[1]) / D)
        
        plt.pause(0.001)
        
    if SAVE and t % SAVE_EVERY == 0 and t > SAVE_AFTER:
        
        jnp.save(ROOT_PATH + f"data/u_{t}.npy", u[:, ::SAVE_DSAMP, ::SAVE_DSAMP].astype(SAVE_FORMAT))
        jnp.save(ROOT_PATH + f"data/rho_{t}.npy", u[:, ::SAVE_DSAMP, ::SAVE_DSAMP].astype(SAVE_FORMAT))
        
        d_save = d_save.at[save_pointer].set(d)
        v_save = v_save.at[save_pointer].set(v)
        a_save = a_save.at[save_pointer].set(a)
        g_save = g_save.at[save_pointer].set(g)
        
        save_pointer += 1
        
if SAVE:
    jnp.savez(ROOT_PATH + "data/dynamics.npz", d=d_save, v=v_save, a=a_save, g=g_save)    
    jnp.save(ROOT_PATH + f"data/f_{t}.npy", f)
