import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from vivsim import lbm, post
import os

# simulation parameters
U0 = 0.5  # velocity * (must < 0.5)
RE_GRID = 20 # Reynolds number based on grid size * (must < 22)
NX = 1000  # number of grid points in x direction *
NY = 1000  # number of grid points in y direction *

# optput parameters
TM = 60000  # number of time steps *
PLOT = False  # whether to plot the results
PLOT_EVERY = 200  # plot every n time steps
PLOT_AFTER = 00  # plot after n time steps
SAVE = True  # whether to save the results
SAVE_EVERY = 100  # save every n time steps
SAVE_AFTER = 19900 # save after n time steps
SAVE_DSAMP = 4 # spatially downsample the output by n
SAVE_FORMAT = jnp.float16  # precision of the output
SAVE_PATH = os.path.dirname(__file__) + "/../output/ldc/"  # path to save the output 

# derived parameters
NU = U0 / RE_GRID  # kinematic viscosity
TAU = 3 * NU + 0.5  # relaxation time
OMEGA = 1 / TAU  # relaxation parameter
OMEGA_MRT = lbm.get_omega_mrt(OMEGA)   # relaxation matrix for MRT

# initialize arrays
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # distribution function
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # equilibrium distribution function
rho = jnp.ones((NX, NY))  # density
u = jnp.zeros((2, NX, NY))  # velocity

# initialize distribution function
f = lbm.get_equilibrum(rho, u, f)

# define the main loop
@jax.jit
def update(f, feq, rho, u):
    
    # Compute equilibrium distribution function
    feq = lbm.get_equilibrum(rho, u, feq)

    # Collision
    f = lbm.collision_mrt(f, feq, OMEGA_MRT)

    # Streaming
    f = lbm.streaming(f)

    # Boundary conditions
    f = lbm.left_wall(f)
    f = lbm.right_wall(f)
    f = lbm.bottom_wall(f)
    f, rho = lbm.top_velocity(f, rho, U0, 0)
        
    # get new macroscopic properties
    rho, u = lbm.get_macroscopic(f, rho, u)
    
    return f, feq, rho, u


if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    plt.subplots(figsize=(5, 4))
    im = plt.imshow(
        post.calculate_velocity(u).T,
        cmap="plasma_r",
        aspect="equal",
        origin="lower",
        )
    plt.colorbar()

for t in tqdm(range(TM)):
    f, feq, rho, u  = update(f, feq, rho, u)
    
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
        im.set_data(post.calculate_velocity(u).T)
        im.autoscale()
        plt.pause(0.001)

    if SAVE and t % SAVE_EVERY == 0 and t > SAVE_AFTER:
        jnp.save(SAVE_PATH + f"u_{t}.npy", u[:, ::SAVE_DSAMP, ::SAVE_DSAMP].astype(SAVE_FORMAT))
