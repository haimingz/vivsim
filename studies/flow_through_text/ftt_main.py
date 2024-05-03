import os
import yaml
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from vivsim import lbm, post

# ----------- load parameters -----------

ROOT_PATH = os.path.dirname(__file__)  + "/tju1895/" # path to save the output 

with open(ROOT_PATH + "parameters.yml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

NX = params["NX"]  # number of grid points in x direction *
NY = params["NY"]  # number of grid points in y direction *
TEXT = params["TEXT"]  # text to be displayed
FONT = params["FONT"]  # font file
SIZE = params["SIZE"]  # font size
U0 = params["U0"]  # velocity * (must < 0.5)
RE_GRID = params["RE_GRID"] # Reynolds number based on grid size * (must < 22)
TM = params["TM"]  # number of time steps *
PLOT = params["PLOT"]  # whether to plot the results
PLOT_EVERY = params["PLOT_EVERY"]  # plot every n time steps
PLOT_AFTER = params["PLOT_AFTER"]  # plot after n time steps
SAVE = params["SAVE"]  # whether to save the results
SAVE_EVERY = params["SAVE_EVERY"]  # save every n time steps
SAVE_AFTER = params["SAVE_AFTER"] # save after n time steps
SAVE_DSAMP = params["SAVE_DSAMP"] # spatially downsample the output by n
if params["SAVE_FORMAT"] == "f32":
    SAVE_FORMAT = jnp.float32  # precision of the output
else:
    SAVE_FORMAT = jnp.float16

# ----------- generate the text mask -----------

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
NU = U0 / RE_GRID  # kinematic viscosity
TAU = 3 * NU + 0.5  # relaxation time
OMEGA = 1 / TAU  # relaxation parameter
OMEGA_MRT = lbm.get_omega_mrt(OMEGA)   # relaxation matrix for MRT

# initialize arrays
rho = jnp.ones((NX, NY))  # density
u = jnp.zeros((2, NX, NY))  # velocity
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # distribution function
f = lbm.get_equilibrum(rho, u, f)
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # equilibrium distribution function

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
    f = lbm.top_outlet(f)
    f, rho = lbm.bottom_velocity(f, rho, 0, U0)
    
    # Obstacle
    f = lbm.obstacle_bounce(f, mask)
    
    # get new macroscopic properties
    rho, u = lbm.get_macroscopic(f, rho, u)

    return f, feq, rho, u

if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    mycmap = mcolors.ListedColormap(['none', 'white'])
    plt.subplots(figsize=(5, 4))
    im = plt.imshow(
        post.calculate_curl(u).T,
        cmap="bwr",
        aspect="equal",
        origin="lower",
        # norm=mpl.colors.CenteredNorm(),
        vmin=-0.03,
        vmax=0.03,
        )
    plt.colorbar()
    plt.imshow(mask.T, cmap=mycmap, alpha=1, origin='lower')
    plt.tight_layout()

for t in tqdm(range(TM)):
    f, feq, rho, u  = update(f, feq, rho, u)

    if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
        im.set_data(post.calculate_curl(u).T)
        # im.autoscale()
        plt.pause(0.001)

    if SAVE and t % SAVE_EVERY == 0 and t > SAVE_AFTER:
        jnp.save(ROOT_PATH + f'data/u_{t}.npy', u[:, ::SAVE_DSAMP, ::SAVE_DSAMP].astype(SAVE_FORMAT))

if SAVE:
    jnp.save(ROOT_PATH + f'data/f_{t}.npy', f)