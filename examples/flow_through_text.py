# In this example, we simulate a 2D flow through a text using LBM.
# The flow is driven by a constant velocity U0 in the y+ direction.

import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from vivsim import lbm, post


# define control parameters
U0 = 0.05  # velocity 
TEXT = 'F'  # text to be displayed
FONT = 'times.ttf'  # font file
SIZE = 100  # font size
RE_GRID = 3  # Reynolds number based on grid size
NX = 500  # number of grid points in x direction
NY = 500  # number of grid points in x direction
TM = 20000   # number of time steps

# plot options
PLOT = True   # whether to plot the results
PLOT_EVERY = 100   # plot every n time steps
PLOT_AFTER = 100   # plot after n time steps

# generate the text mask
img = Image.new("L", (NX, NY), 0)
draw = ImageDraw.Draw(img)
draw.text((NX // 2, NY // 2), 
          TEXT, 
          font=ImageFont.truetype(FONT, SIZE), 
          fill=255, 
          anchor='mm',
          )

mask = np.array(img)[::-1].T
mask = (mask > 0).astype(np.bool_)

# derived parameters
NU = U0 / RE_GRID   # kinematic viscosity
TAU = 3 * NU + 0.5   # relaxation time
OMEGA = 1 / TAU   # relaxation parameter
OMEGA_MRT = lbm.get_omega_mrt(OMEGA)    # relaxation matrix for MRT

# create empty arrays
rho = jnp.ones((NX, NY), dtype=jnp.float32)   # density
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)   # velocity
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)   # distribution function
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)   # equilibrium distribution function

# initialize
f = lbm.get_equilibrum(rho, u, f)

# define main loop
@jax.jit
def update(f, feq, rho, u):

    # Compute equilibrium distribution function
    feq = lbm.get_equilibrum(rho, u, feq)

     # Collision
    f = lbm.collision_mrt(f, feq, OMEGA_MRT)

     # Streaming
    f = lbm.streaming(f)

     # Boundary conditions
    f = lbm.top_outlet_simple(f)
    f, rho = lbm.bottom_velocity_nebb(f, rho, 0, U0)
    
     # Obstacle
    f = lbm.obj_noslip_bb(f, mask)
    
     # get new macroscopic properties
    rho, u = lbm.get_macroscopic(f, rho, u)

    return f, feq, rho, u

# create the plot template
if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    mycmap = mcolors.ListedColormap(['none', 'white'])
    plt.subplots(figsize=(5, 4))
    im = plt.imshow(
        post.calculate_velocity(u).T,
        cmap="viridis",
        aspect="equal",
        origin="lower",
        vmax=0.3,
        vmin=0.0,
        extent=[0, NX, 0, NY],
        )
    plt.colorbar()
    plt.imshow(mask.T, cmap=mycmap, alpha=1, origin='lower')
    plt.tight_layout()

# start simulation   
for t in tqdm(range(TM)):
    f, feq, rho, u  = update(f, feq, rho, u)

    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
        im.set_data(post.calculate_velocity(u).T)
        # im.autoscale()
        plt.pause(0.001)
