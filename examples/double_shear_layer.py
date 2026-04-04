"""
2D double shear layer roll-up on a periodic domain.

This classic benchmark starts from two thin, oppositely signed shear layers
with a small transverse perturbation. As the simulation evolves, the layers
become unstable and roll up into vortical structures.
"""

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from vivsim import lbm, post


# ====================== Configuration ======================

NX = 128
NY = 128
TM = 10000
PLOT = True
PLOT_EVERY = 20

U0 = 0.04
NU = 0.002
OMEGA = lbm.get_omega(NU)

SHEAR_STEEPNESS = 80.0
PERTURBATION = 0.05 * U0

X = (jnp.arange(NX, dtype=jnp.float32) + 0.5) / NX
Y = (jnp.arange(NY, dtype=jnp.float32) + 0.5) / NY
X_GRID = X[:, None]
Y_GRID = Y[None, :]


def initial_velocity():
    upper_layer = jnp.tanh(SHEAR_STEEPNESS * (Y_GRID - 0.25))
    lower_layer = jnp.tanh(SHEAR_STEEPNESS * (0.75 - Y_GRID))
    ux = U0 * jnp.where(Y_GRID <= 0.5, upper_layer, lower_layer)
    ux = jnp.broadcast_to(ux, (NX, NY))
    uy = PERTURBATION * jnp.sin(2 * jnp.pi * X_GRID)
    uy = jnp.broadcast_to(uy, (NX, NY))
    return jnp.stack((ux, uy))


@jax.jit
def update(f):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)
    f = lbm.streaming(f)
    return f


def main():
    rho0 = jnp.ones((NX, NY), dtype=jnp.float32)
    u0 = initial_velocity()
    f = lbm.get_equilibrium(rho0, u0)
    vort0 = post.calculate_vorticity(u0)

    if PLOT:
        mpl.rcParams["figure.raise_window"] = False
        vort_lim = float(jnp.max(jnp.abs(vort0)))
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(
            np.asarray(vort0.T),
            origin="lower",
            cmap="RdBu_r",
            extent=[0, NX - 1, 0, NY - 1],
            vmin=-vort_lim,
            vmax=vort_lim,
        )
        ax.set_title("Double shear layer, step = 0")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, label="$\\omega_z$")
        fig.tight_layout()

    for step in tqdm(range(1, TM + 1), desc="Double shear layer"):
        f = update(f)
        should_plot = PLOT and step % PLOT_EVERY == 0

        if should_plot:
            _, u = lbm.get_macroscopic(f)
            vort = post.calculate_vorticity(u)

        if should_plot:
            im.set_data(np.asarray(vort.T))
            im.set_clim(vmin=-float(jnp.max(jnp.abs(vort))), vmax=float(jnp.max(jnp.abs(vort))))
            ax.set_title(f"Double shear layer, step = {step}")
            plt.pause(0.001)

    if PLOT:
        plt.show()


if __name__ == "__main__":
    main()
