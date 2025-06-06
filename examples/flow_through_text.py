# In this example, we simulate a 2D flow through a text using LBM.
# The flow is driven by a constant velocity U0 in the y+ direction.

import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
from vivsim import lbm, post, mrt


# ====================== plot options ======================

PLOT = True   # whether to plot the results during simulation
PLOT_EVERY = 100   # plot every n time steps
PLOT_AFTER = 100   # plot after n time steps


# =================== define fluid parameters ===================

# fluid parameters
NU = 0.005          # kinematic viscosity
RE_GRID = 5         # Reynolds number based on grid size
U0 = RE_GRID * NU   # velocity

# LBM parameters
TAU = 3 * NU + 0.5  # relaxation time
OMEGA = 1 / TAU     # relaxation parameter

# MRT parameters
MRT_TRANS = mrt.get_trans_matrix()
MRT_RELAX = mrt.get_relax_matrix(OMEGA)
MRT_COL_LEFT = mrt.get_collision_left_matrix(MRT_TRANS, MRT_RELAX)


# =================== setup computation domain ===================

NX = 500       # number of grid points in x direction
NY = 500       # number of grid points in x direction
TM = 50000      # number of time steps

rho = jnp.ones((NX, NY), dtype=jnp.float32)      # density
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)    # velocity
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)    # distribution function
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # equilibrium distribution function


# =================== define text obstacle ===================

TEXT = 'VIVSIM'  # text to be displayed
FONT_FAMILY = 'DejaVuSans.ttf'  # font file
FONT_SIZE = NX // 5  # font size

# generate the text mask
img = Image.new("L", (NX, NY), 0)
draw = ImageDraw.Draw(img)
draw.text((NX // 2, NY // 2), 
          TEXT, 
          font=ImageFont.truetype(FONT_FAMILY, FONT_SIZE), 
          fill=255, 
          anchor='mm',
          )

MASK = jnp.array(img).astype(bool)[::-1].T

# =================== initialize ===================

u = u.at[1].set(U0)
f = lbm.get_equilibrium(rho, u)

# =================== define calculation routine ===================

@jax.jit
def update(f):
      
    # Collision
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = mrt.collision(f, feq, MRT_COL_LEFT)

    # Streaming
    f = lbm.streaming(f)

    # Boundary conditions
    f = lbm.outlet_boundary(f, loc='top')
    f = lbm.velocity_boundary(f, 0, U0, loc='bottom')
    
    # Obstacle
    f = lbm.noslip_obstacle(f, MASK)
    
    return f, feq, rho, u


# =============== create plot template ================

if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    mycmap = mcolors.ListedColormap(['none', 'white']) # colormap for the text
    plt.subplots(figsize=(5, 4))
    im = plt.imshow(
        post.calculate_velocity_magnitude(u).T,
        cmap="viridis",
        aspect="equal",
        origin="lower",
        vmax=U0 * 5,
        vmin=0.0,
        extent=[0, NX, 0, NY],
        )
    plt.colorbar()
    plt.imshow(MASK.T, cmap=mycmap, alpha=1, origin='lower') # plot the text over the fluid
    plt.tight_layout()


# =============== start simulation ===============

for t in tqdm(range(TM)):
    f, feq, rho, u  = update(f)

    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
        im.set_data(post.calculate_velocity_magnitude(u).T)
        plt.pause(0.001)
