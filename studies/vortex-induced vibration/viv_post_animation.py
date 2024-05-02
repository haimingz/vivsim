import os 
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from vivsim import post

ROOT_PATH = os.path.dirname(__file__) + "/case_demo/" # path to save the output 

# ----------------------- load parameters from config file -----------------------

with open(ROOT_PATH + "parameters.yml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

X_OBJ = params["X_OBJ"]  # y coordinate of the cylinder *
Y_OBJ = params["Y_OBJ"]  # x coordinate of the cylinder *
D = params["D"]  # diameter *
TM = params["TM"]  # number of time steps *
NX = params["NX"]  # number of grid points in x direction *
NY = params["NY"]  # number of grid points in y direction *
SAVE_DSAMP = params["SAVE_DSAMP"] # spatially downsample the output by n
SAVE_EVERY = params["SAVE_EVERY"]  # save every n time steps
SAVE_AFTER = params["SAVE_AFTER"] # save after n time steps

# ----------------------- generate first plot -----------------------

u = np.load(ROOT_PATH + f'data/u_100.npy')
dynamics = np.load(ROOT_PATH + f'data/dynamics.npz')
d = dynamics['d'][0]
curl = post.calculate_curl(u)

fig, ax = plt.subplots(figsize=(7.5, 3))
im = plt.imshow(
    curl.T,
    cmap="bwr",
    aspect="equal",
    origin="lower",
    extent=[0, NX, 0, NY],
    # norm=mpl.colors.CenteredNorm(),
    vmin=-0.04,
    vmax=0.04,
    )
plt.colorbar()
circle = plt.Circle(((X_OBJ + d[0]), (Y_OBJ + d[1])), D / 2, 
                        edgecolor='black', linewidth=0.3,
                        facecolor='white', fill=True)
plt.gca().add_artist(circle)
plt.axvline(X_OBJ, color="k", linestyle="--", linewidth=0.5)
plt.axhline(Y_OBJ, color="k", linestyle="--", linewidth=0.5)
plt.xlim(100, NX)
plt.ylim(30, NY - 30)
plt.xticks([])
plt.yticks([])
plt.tight_layout()

# ----------------------- define animation function -----------------------

def update(i):
    u = np.load(ROOT_PATH + f'data/u_{i}.npy')
    d = dynamics['d'][int((i - SAVE_AFTER) // SAVE_EVERY - 1)]
    curl = post.calculate_curl(u)
    im.set_data(curl.T)
    # im.autoscale()
    circle.center = ((X_OBJ + d[0]), (Y_OBJ + d[1]))
    return im

# ----------------------- generate animation -----------------------

ani = FuncAnimation(fig, update, frames=range(SAVE_AFTER + SAVE_EVERY, int(TM * 0.8), SAVE_EVERY * 2), interval=1)
ani.save(ROOT_PATH + 'viv.gif', fps=60)
# plt.show()