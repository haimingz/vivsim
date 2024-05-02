import os
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from vivsim import lbm, post


ROOT_PATH = os.path.dirname(__file__)  + "/" # path to save the output 

# ----------- generate the text mask -----------

NX, NY = 1000,1000

img = Image.new("L", (NX, NY), 0)
draw = ImageDraw.Draw(img)
draw.text((NX // 2, NY // 2), 
          "TJU", 
          font=ImageFont.truetype("arial.ttf", 200), 
          fill=255, 
          anchor='mm',
          )

mask = np.array(img)[::-1].T
mask = (mask > 0).astype(np.bool_)

# plt.figure(figsize=(10, 5))
# plt.imshow(mask, cmap='gray', origin='lower')
# plt.show()

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

U0 = 0.05 # velocity * (must < 0.5)
RE_GRID = 3 # Reynolds number based on grid size
NU = U0 / RE_GRID  # kinematic viscosity
# NU = 0.1
TAU = 3 * NU + 0.5  # relaxation time
OMEGA = 1 / TAU  # relaxation parameter
OMEGA_MRT = lbm.get_omega_mrt(OMEGA)   # relaxation matrix for MRT

TM = 20000  # number of time steps *
PLOT = True  # whether to plot the results
PLOT_EVERY = 100  # plot every n time steps
PLOT_AFTER = 00  # plot after n time steps
SAVE = False  # whether to save the results
SAVE_EVERY = 100  # save every n time steps
SAVE_AFTER = 0  # save after n time steps
SAVE_DSAMP = 1  # spatially downsample the output by n
SAVE_FORMAT = jnp.float32  # precision of the output

# initialize arrays
rho = jnp.ones((NX, NY))  # density
u = jnp.zeros((2, NX, NY))  # velocity
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # distribution function
f = lbm.get_equilibrum(rho, u, f)
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # equilibrium distribution function

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
        jnp.save(ROOT_PATH + f'data/u_{t}.npy', u)

if SAVE:
    jnp.save(ROOT_PATH + f'data/f_{t}.npy', f)