import os 
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from vivsim import post
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont

ROOT_PATH = os.path.dirname(__file__) + "/"  # path to save the output 

NX, NY = 1000,1000
TM = 20000  # number of time steps *
SAVE_EVERY = 100  # save every n time steps
SAVE_AFTER = 0 # save after n time steps

img = Image.new("L", (NX, NY), 0)
draw = ImageDraw.Draw(img)
draw.text((NX // 2, NY // 2), 
          "VIVSIM", 
          font=ImageFont.truetype("arial.ttf", 200), 
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

ani = FuncAnimation(fig, update, frames=range(SAVE_AFTER + SAVE_EVERY * 10, TM,SAVE_EVERY * 2), interval=1)
ani.save(ROOT_PATH + 'fot.gif', fps=60)
# plt.show()