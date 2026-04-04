"""
Plane Couette flow in a 2D channel driven by a moving lid.

This classic benchmark complements the existing cavity and Poiseuille examples:
the streamwise direction is periodic, the lower wall is stationary, and the
upper wall moves with a constant velocity. The steady analytical solution is a
linear velocity profile.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from vivsim import lbm


# ====================== Configuration ======================

NX = 96
NY = 32
TM = 12000

U0 = 0.08
NU = 0.10
OMEGA = lbm.get_omega(NU)


def create_initial_state():
    rho = jnp.ones((NX, NY), dtype=jnp.float32)
    u = jnp.zeros((2, NX, NY), dtype=jnp.float32)
    f = lbm.get_equilibrium(rho, u)
    return f


@jax.jit
def update(f):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_bgk(f, feq, OMEGA)
    f = lbm.streaming(f)
    f = lbm.boundary_nee(f, loc="bottom")
    f = lbm.boundary_nee(f, loc="top", ux_wall=U0)
    return f


def main():
    f = create_initial_state()

    for _ in tqdm(range(TM), desc="Couette flow"):
        f = update(f)

    rho, u = lbm.get_macroscopic(f)
    ux = u[0]

    y = jnp.arange(NY, dtype=jnp.float32)
    ux_true = U0 * y / (NY - 1)
    ux_mean = jnp.mean(ux, axis=0)
    ux_true_2d = jnp.broadcast_to(ux_true, (NX, NY))

    rel_l2 = jnp.linalg.norm(ux - ux_true_2d) / jnp.linalg.norm(ux_true_2d)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    im = axes[0].imshow(
        np.asarray(ux.T),
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[0, NX - 1, 0, NY - 1],
    )
    axes[0].set_title("Steady x-velocity")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes[0], label="$u_x$")

    axes[1].plot(np.asarray(ux_true), np.asarray(y), label="Analytical", linewidth=2)
    axes[1].plot(np.asarray(ux_mean), np.asarray(y), "o", ms=3, label="LBM")
    axes[1].set_title(f"Profile error = {float(rel_l2):.2e}")
    axes[1].set_xlabel("$u_x$")
    axes[1].set_ylabel("y")
    axes[1].grid(alpha=0.3, linestyle=":")
    axes[1].legend(frameon=False)

    fig.suptitle("2D plane Couette flow")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
