import os 
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from vivsim import post
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont

ROOT_PATH = os.path.dirname(__file__) + "/tju1895/"  # path to save the output 

with open(ROOT_PATH + "parameters.yml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
    
NX = params["NX"]  # number of grid points in x direction *
NY = params["NY"]  # number of grid points in y direction *
TEXT = params["TEXT"]  # text to be displayed
FONT = params["FONT"]  # font file
SIZE = params["SIZE"]  # font size
TM = params["TM"]  # number of time steps *
SAVE_EVERY = params["SAVE_EVERY"]  # save every n time steps
SAVE_AFTER = params["SAVE_AFTER"] # save after n time steps
SAVE_DSAMP = params["SAVE_DSAMP"] # spatially downsample the output by n

# ----------- re-generate the text mask -----------

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

# initialize the plot
mycmap = mcolors.ListedColormap(['none', 'white'])

u = np.load(ROOT_PATH + f'data/u_100.npy')
fig, ax = plt.subplots(figsize=(5, 4))
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
plt.xticks([])
plt.yticks([])

def update(i):
    u = np.load(ROOT_PATH + f'data/u_{i}.npy')
    im.set_data(post.calculate_velocity(u).T)
    # im.autoscale()
    return im

ani = FuncAnimation(fig, update, frames=range(3000, TM, SAVE_EVERY * 2), interval=1)
ani.save(ROOT_PATH + f'ftt_{TEXT}.gif', fps=60)
ani.save(ROOT_PATH + f'ftt_{TEXT}.mp4', fps=30)
plt.show()