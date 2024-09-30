# In this example, we simulate a 2D lid-driven cavity flow using LBM.
# The lid is moving at a constant velocity U0 in the x direction.


import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from vivsim import lbm, post

import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    # '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)
# jax.config.update('jax_platform_name', 'cpu')

# define control parameters
U0 = 0.5  # velocity (must < 0.5 for stability)
RE_GRID = 20 # Reynolds number based on grid size (must < 22 for stability)
NX = 200  # number of grid points in x direction
NY = 200  # number of grid points in y direction
TM = 50000  # number of time steps

# plot options
PLOT = True  # whether to plot the results
PLOT_EVERY = 200  # plot every n time steps
PLOT_AFTER = 00  # plot after n time steps

# derived parameters for LBM
NU = U0 / RE_GRID  # kinematic viscosity
TAU = 3 * NU + 0.5  # relaxation time
OMEGA = 1 / TAU  # relaxation parameter
OMEGA_MRT = lbm.get_omega_mrt(OMEGA)   # relaxation matrix for MRT
    
# create empty arrays
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # distribution function
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # equilibrium distribution function
rho = jnp.ones((NX, NY), dtype=jnp.float32)  # density
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)  # velocity

# initialize
f = lbm.get_equilibrum(rho, u, f)

# define the update function for each time step
def update(f, feq, rho, u):
    
    # Collision
    feq = lbm.get_equilibrum(rho, u, feq)
    f = lbm.collision_mrt(f, feq, OMEGA_MRT)

    # Streaming
    f = lbm.streaming(f)

    # Boundary conditions
    f = lbm.left_soild(f)
    f = lbm.right_soild(f)
    f = lbm.bottom_soild(f)
    f, rho = lbm.top_velocity(f, rho, U0, 0)
        
    # get new macroscopic properties
    rho, u = lbm.get_macroscopic(f, rho, u)
    
    return f, feq, rho, u

# create the plot template
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

# start simulation 
update_jitted = jax.jit(update)

for t in tqdm(range(TM)):   
    f, feq, rho, u  = update_jitted(f, feq, rho, u)
    
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
        im.set_data(post.calculate_velocity(u).T)
        im.autoscale()
        plt.pause(0.001)