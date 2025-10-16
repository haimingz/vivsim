"""
2D lid-driven cavity flow using LBM.

This example can be used to test the stability of different collision models
and boundary conditions. Genrally, the non-equilibrium extrapolation (NEE) scheme
is more stable than the non-equilibrium bounce-back (NEBB) scheme. Adopting the
KBC collision model also helps improve stability at the boundaries.
"""



import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from vivsim import lbm, post

# ====================== plot options ======================

PLOT = True                 # whether to plot the results during simulation
PLOT_EVERY = 500            # plot every n time steps

# =================== define fluid parameters ===================

U0 = 0.3                    # velocity of the moving lid
RE_GRID = 30                # Reynolds number based on grid size

NU = U0 / RE_GRID           # kinematic viscosity
OMEGA = lbm.get_omega(NU)   # relaxation parameter

# =================== setup computation domain ===================

NX = 1000                   # number of grid points in x direction
NY = 1000                   # number of grid points in y direction
TM = 80000                  # number of time steps

# =================== initialize ===================

rho = jnp.ones((NX, NY))    
u = jnp.zeros((2, NX, NY))  
f = lbm.get_equilibrium(rho, u)

# =================== define the update function ===================

@jax.jit
def update(f):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_bgk(f, feq, OMEGA)
    f = lbm.streaming(f)
    f = lbm.boundary_nee(f, loc='top', ux_wall=U0)
    f = lbm.boundary_nee(f, loc='left')
    f = lbm.boundary_nee(f, loc='right')
    f = lbm.boundary_nee(f, loc='bottom')
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
    
    if PLOT and t % PLOT_EVERY == 0:
        im.set_data(post.calculate_velocity_magnitude(u).T)
        im.autoscale()
        plt.pause(0.001)