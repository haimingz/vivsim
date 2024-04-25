import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from vivsim import post

SAVE_PATH = os.path.dirname(__file__) + "/../output/ldc/"  # path to save the output 

# initialize the plot
u = np.load(SAVE_PATH + f'u_100.npy')
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
    u = np.load(SAVE_PATH + f'u_{i}.npy')
    im.set_data(post.calculate_velocity(u).T)
    return im

ani = FuncAnimation(fig, update, frames=range(100,60000,300), interval=1)
# ani.save('ldc.gif', writer='imagemagick', fps=60)
plt.show()