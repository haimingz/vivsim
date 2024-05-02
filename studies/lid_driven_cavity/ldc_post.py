import os 
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from vivsim import post

ROOT_PATH = os.path.dirname(__file__) + "/case_demo/"  # path to save the output 

with open(ROOT_PATH + "parameters.yml") as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

TM = params["TM"]  # number of time steps *
SAVE_EVERY = params["SAVE_EVERY"]  # save every n time steps
SAVE_AFTER = params["SAVE_AFTER"] # save after n time steps

# initialize the plot
u = np.load(ROOT_PATH + f'data/u_100.npy')
fig, ax = plt.subplots(figsize=(5, 4))
im = plt.imshow(
    post.calculate_velocity(u).T,
    cmap="plasma_r",
    aspect="equal",
    origin="lower",
    vmax=0.4,
    vmin=0.0,
    )
plt.colorbar()
plt.xticks([])
plt.yticks([])

def update(i):
    u = np.load(ROOT_PATH + f'data/u_{i}.npy')
    im.set_data(post.calculate_velocity(u).T)
    return im

ani = FuncAnimation(fig, update, frames=range(SAVE_AFTER + SAVE_EVERY,TM,SAVE_EVERY * 5), interval=1)
ani.save('ldc.gif', writer='imagemagick', fps=60)
# plt.show()