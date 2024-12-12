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
from vivsim import lbm, post, mrt

# define the text obstacle in the fluid
TEXT = 'VIVSIM'  # text to be displayed
FONT_FAMILY = 'times.ttf'  # font file
FONT_SIZE = 100  # font size

# domain size and time steps
NX = 500  # number of grid points in x direction
NY = 500  # number of grid points in x direction
TM = 50000   # number of time steps

# fluid properties
NU = 0.005   # kinematic viscosity
RE_GRID = 5  # Reynolds number based on grid size
U0 = RE_GRID * NU # velocity 

# plot options
PLOT = True   # whether to plot the results
PLOT_EVERY = 100   # plot every n time steps
PLOT_AFTER = 100   # plot after n time steps

# lbm parameters
TAU = 3 * NU + 0.5   # relaxation time
OMEGA = 1 / TAU   # relaxation parameter
MRT_TRANS = mrt.get_trans_matrix()
MRT_RELAX = mrt.get_relax_matrix(OMEGA)
MRT_COL_LEFT = mrt.get_collision_left_matrix(MRT_TRANS, MRT_RELAX)

# create empty arrays
rho = jnp.ones((NX, NY), dtype=jnp.float32)   # density
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)   # velocity
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)   # distribution function
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)   # equilibrium distribution function

# generate the text mask
img = Image.new("L", (NX, NY), 0)
draw = ImageDraw.Draw(img)
draw.text((NX // 2, NY // 2), 
          TEXT, 
          font=ImageFont.truetype(FONT_FAMILY, FONT_SIZE), 
          fill=255, 
          anchor='mm',
          )

mask = np.array(img)[::-1].T
mask = (mask > 0).astype(np.bool_)

# initialize fluid distribution function
f = lbm.get_equilibrium(rho, u, f)

# define main loop
@jax.jit
def update(f, feq, rho, u):

    # Compute equilibrium distribution function
    feq = lbm.get_equilibrium(rho, u, feq)

    # Collision
    f = mrt.collision(f, feq, MRT_COL_LEFT)

    # Streaming
    f = lbm.streaming(f)

    # Boundary conditions
    f = lbm.top_outflow(f)
    f = lbm.bottom_velocity(f, 0, U0)
    
    # Obstacle
    f = lbm.obstacle_noslip(f, mask)
    
    # get new macroscopic properties
    rho, u = lbm.get_macroscopic(f, rho, u)

    return f, feq, rho, u

# create the plot template
if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    mycmap = mcolors.ListedColormap(['none', 'white']) # colormap for the text
    plt.subplots(figsize=(5, 4))
    im = plt.imshow(
        post.calculate_velocity(u).T,
        cmap="viridis",
        aspect="equal",
        origin="lower",
        vmax=U0 * 5,
        vmin=0.0,
        extent=[0, NX, 0, NY],
        )
    plt.colorbar()
    plt.imshow(mask.T, cmap=mycmap, alpha=1, origin='lower') # plot the text over the fluid
    plt.tight_layout()

# start simulation   
for t in tqdm(range(TM)):
    f, feq, rho, u  = update(f, feq, rho, u)

    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
        im.set_data(post.calculate_velocity(u).T)
        # im.autoscale()
        plt.pause(0.001)
