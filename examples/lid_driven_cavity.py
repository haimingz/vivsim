# In this example, we simulate a 2D lid-driven cavity flow using LBM.
# The lid is moving at a constant velocity U0 in the x direction.


import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from vivsim import lbm, post, mrt

# ====================== plot options ======================

PLOT = True   # whether to plot the results during simulation
PLOT_EVERY = 200   # plot every n time steps
PLOT_AFTER = 100   # plot after n time steps


# =================== define fluid parameters ===================

# fluid parameters
U0 = 0.5  # velocity (must < 0.5 for stability)
RE_GRID = 20 # Reynolds number based on grid size (must < 22 for stability)
NU = U0 / RE_GRID  # kinematic viscosity

# LBM parameters
TAU = 3 * NU + 0.5  # relaxation time
OMEGA = 1 / TAU  # relaxation parameter

# MRT parameters
MRT_TRANS = mrt.get_trans_matrix()
MRT_RELAX = mrt.get_relax_matrix(OMEGA)
MRT_COL_LEFT = mrt.get_collision_left_matrix(MRT_TRANS, MRT_RELAX)  

# =================== setup computation domain ===================

NX = 1000  # number of grid points in x direction
NY = 1000  # number of grid points in y direction
TM = 50000  # number of time steps

rho = jnp.ones((NX, NY), dtype=jnp.float32)      # density
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)    # velocity
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)    # distribution function
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # equilibrium distribution function


# =================== initialize ===================

f = lbm.get_equilibrium(rho, u)

# =================== define the update function ===================

@jax.jit
def update(f):
    
    # Collision
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = mrt.collision(f, feq, MRT_COL_LEFT)

    # Streaming
    f = lbm.streaming(f)

    # Boundary conditions
    f = lbm.noslip_boundary(f, loc='left')
    f = lbm.noslip_boundary(f, loc='right')
    f = lbm.noslip_boundary(f, loc='bottom')
    f = lbm.velocity_boundary(f, U0, 0, loc='top')
    
    return f, feq, rho, u


# =================== create the plot template ===================

if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    plt.subplots(figsize=(5, 4))
    im = plt.imshow(
        post.calculate_velocity_magnitude(u).T,
        cmap="plasma_r",
        aspect="equal",
        origin="lower",
        )
    plt.colorbar()


# =================== start simulation ===================

for t in tqdm(range(TM)):   
    f, feq, rho, u  = update(f)
    
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
        im.set_data(post.calculate_velocity_magnitude(u).T)
        im.autoscale()
        plt.pause(0.001)