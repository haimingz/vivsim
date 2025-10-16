# 2D flow through a text using LBM.
# Just for fun and testing the boundary condition implementation.

import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
from vivsim import lbm, post



# ====================== plot options ======================

PLOT = True                     # whether to plot the results during simulation
PLOT_EVERY = 200                # plot every n time steps


# =================== define fluid parameters ===================

U0 = 0.05                        # kinematic viscosity
RE_GRID = 20                   # Reynolds number based on grid size

NU = U0 / RE_GRID               # kinematic viscosity
OMEGA = lbm.get_omega(NU)       # relaxation parameter

# =================== setup computation domain ===================

NX = 500                        # number of grid points in x direction
NY = 500                        # number of grid points in y direction
TM = 80000                      # number of time steps

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

    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)
    f = lbm.streaming(f)
    f = lbm.boundary_nee(f, loc='bottom', uy_wall=U0)
    f = lbm.boundary_equilibrium(f, loc='top', uy_wall=U0)
    f = lbm.obstacle_bounce_back(f, MASK)
    
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
        vmax=U0 * 4,
        vmin=0.0,
        extent=[0, NX, 0, NY],
        )
    plt.colorbar()
    plt.imshow(MASK.T, cmap=mycmap, alpha=1, origin='lower') # plot the text over the fluid
    plt.tight_layout()


# =============== start simulation ===============

for t in tqdm(range(TM)):
    f, feq, rho, u  = update(f)

    if PLOT and t % PLOT_EVERY == 0:
        im.set_data(post.calculate_velocity_magnitude(u).T)
        plt.pause(0.001)
