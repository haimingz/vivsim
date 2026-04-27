r"""
2D flow through text-shaped obstacles using LBM

This fun example renders text as solid obstacles in a 2D flow domain and
simulates the flow field around them using the bounce-back boundary condition.

"""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import random
from tqdm import tqdm

from vivsim import lbm, post


# ========================== VISUALIZATION PARAMETERS ====================

PLOT = True    # Enable chunked live plotting
CHUNK_STEPS = 200


# ========================== GEOMETRY =======================

NX = 500      # Number of grid points in x-direction
NY = 500      # Number of grid points in y-direction

# Text obstacle
TEXT = "VIVSIM"                    # Text to render as obstacle
FONT_SIZE = NX // 5               # Font size in pixels


img = Image.new("L", (NX, NY), 0)
draw = ImageDraw.Draw(img)
draw.text(
    (NX // 2, NY // 2), TEXT,
    font=ImageFont.truetype(random.choice(fm.findSystemFonts(fontext='ttf')), FONT_SIZE),
    fill=255, anchor="mm",
)
MASK = jnp.array(img).astype(bool)[::-1].T  # Boolean obstacle mask


# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.02         # Inlet velocity (upward)
RE_GRID = 10       # Reynolds number based on grid size

# Derived physical parameters
NU = U0 / RE_GRID              # Kinematic viscosity
OMEGA = lbm.get_omega(NU)      # Relaxation parameter
TM = 100_000         # Total simulation timesteps


# ======================= INITIALIZE VARIABLES ====================

rho = jnp.ones((NX, NY))           # Fluid density
u = jnp.zeros((2, NX, NY))         # Fluid velocity
u = u.at[1].set(U0)                # Set initial y-velocity to U0
f = lbm.get_equilibrium(rho, u)    # Initial distribution function


# ======================= SIMULATION ROUTINE =====================

def update_step(f):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)
    f = lbm.streaming(f)
    f = lbm.boundary_nee(f, loc="bottom", uy_wall=U0)
    f = lbm.boundary_equilibrium(f, loc="top", uy_wall=U0)
    f = lbm.obstacle_bounce_back(f, MASK)
    return f


@partial(jax.jit, static_argnums=1, donate_argnums=0)
def update_chunk(carry, n_steps):
    def step(carry, _):
        (f,) = carry
        f = update_step(f)
        return (f,), None
    return jax.lax.scan(step, carry, None, length=n_steps)


@jax.jit
def get_plot_image(f):
    _, u = lbm.get_macroscopic(f)
    return post.velocity_magnitude(u).T


# ======================= VISUALIZATION SETUP ====================

mpl.rcParams["figure.raise_window"] = False

if PLOT:
    mycmap = mcolors.ListedColormap(["none", "white"])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        get_plot_image(f),
        cmap="viridis", aspect="equal", origin="lower",
        vmax=U0 * 4, vmin=0.0,
        extent=[0, NX, 0, NY],
    )
    fig.colorbar(im, ax=ax, label="$|u|$")
    ax.imshow(MASK.T, cmap=mycmap, alpha=1, origin="lower",
              extent=[0, NX, 0, NY])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()


# ========================== RUN SIMULATION ==========================

chunk_sizes = [CHUNK_STEPS] * (TM // CHUNK_STEPS)
if TM % CHUNK_STEPS:
    chunk_sizes.append(TM % CHUNK_STEPS)

with tqdm(total=TM, unit="step") as pbar:
    for n_steps in chunk_sizes:
        (f,), _ = update_chunk((f,), n_steps)
        jax.block_until_ready(f)
        pbar.update(n_steps)

        if PLOT:
            im.set_data(get_plot_image(f))
            plt.pause(0.001)