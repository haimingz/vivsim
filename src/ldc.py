import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from vivsim import lbm, post


U0 = 0.5  # velocity *
NX = 512  # number of grid points in x direction *
NY = 512  # number of grid points in y direction *
RE = 5000  # Reynolds number *
NU = U0 * NX / RE  # kinematic viscosity

TM = 80000

# collision operators

TAU = 3 * NU + 0.5  # relaxation time
OMEGA = 1 / TAU  # relaxation parameter
OMEGA_MRT = lbm.get_omega_mrt(OMEGA)   # relaxation matrix for MRT


# main loop
@jax.jit
def update(f, feq, rho, u):
    """Update distribution functions for one time step"""
    
    # Compute equilibrium
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
        
    # new macroscopic properties
    rho, u = lbm.get_macroscopic(f, rho, u)
    
    return f, feq, rho, u, 


# ----------------- initialize properties -----------------
PLOT = True
PLOT_EVERY = 50
PLOT_AFTER = 00

# macroscopic properties
rho = jnp.ones((NX, NY))
u = jnp.zeros((2, NX, NY))
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # distribution functions
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # equilibrium distribution functions

# mesoscopics population
f = lbm.get_equilibrum(rho, u, feq)

if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    plt.figure(figsize=(8, 4))
    
for t in tqdm(range(TM)):
    f, feq, rho, u  = update(f, feq, rho, u)
    
    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
        plt.clf()
        plt.imshow(
            post.calculate_velocity(u).T,
            cmap="Blues",
            aspect="equal",
            origin="lower",
        )
        plt.colorbar()
        plt.pause(0.001)


