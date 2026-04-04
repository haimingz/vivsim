"""
2D Taylor-Green vortex decay on a periodic domain.

This is a classic analytical benchmark for periodic LBM solvers. The initial
velocity field is divergence-free, and the exact solution decays exponentially
with a rate controlled by the viscosity and wave number.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from vivsim import lbm, post


# ====================== Configuration ======================

NX = 64
NY = 64
TM = 4000
SAMPLE_EVERY = 100

U0 = 0.02
NU = 0.02
OMEGA = lbm.get_omega(NU)

KX = 2 * jnp.pi / NX
KY = 2 * jnp.pi / NY
K2 = KX ** 2 + KY ** 2

X = jnp.arange(NX, dtype=jnp.float32)[:, None]
Y = jnp.arange(NY, dtype=jnp.float32)[None, :]


def exact_velocity(t):
    decay = jnp.exp(-NU * K2 * t)
    ux = U0 * jnp.sin(KX * X) * jnp.cos(KY * Y) * decay
    uy = -U0 * jnp.cos(KX * X) * jnp.sin(KY * Y) * decay
    return jnp.stack((ux, uy))


def kinetic_energy(u):
    return 0.5 * jnp.mean(jnp.sum(u ** 2, axis=0))


@jax.jit
def update(f):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_bgk(f, feq, OMEGA)
    f = lbm.streaming(f)
    return f


def main():
    rho0 = jnp.ones((NX, NY), dtype=jnp.float32)
    u0 = exact_velocity(0.0)
    f = lbm.get_equilibrium(rho0, u0)

    sample_steps = np.arange(0, TM + 1, SAMPLE_EVERY)
    energy_hist = np.zeros_like(sample_steps, dtype=np.float64)
    energy_hist[0] = float(kinetic_energy(u0))

    sample_idx = 1
    for step in tqdm(range(1, TM + 1), desc="Taylor-Green vortex"):
        f = update(f)

        if step % SAMPLE_EVERY == 0:
            _, u = lbm.get_macroscopic(f)
            energy_hist[sample_idx] = float(kinetic_energy(u))
            sample_idx += 1

    _, u = lbm.get_macroscopic(f)
    u_true = exact_velocity(float(TM))
    vorticity = post.calculate_vorticity(u)

    rel_l2 = jnp.linalg.norm(u - u_true) / jnp.linalg.norm(u_true)
    energy_true = energy_hist[0] * np.exp(-2 * float(NU * K2) * sample_steps)

    slice_y = NY // 8
    x = np.arange(NX)
    grid_x, grid_y = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    quiver_stride = max(NX // 16, 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    axes[0].plot(sample_steps, energy_true, label="Analytical", linewidth=2)
    axes[0].plot(sample_steps, energy_hist, "o", ms=3, label="LBM")
    axes[0].set_title("Kinetic energy decay")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Energy")
    axes[0].grid(alpha=0.3, linestyle=":")
    axes[0].legend(frameon=False)

    axes[1].plot(x, np.asarray(u_true[0, :, slice_y]), label="Analytical", linewidth=2)
    axes[1].plot(x, np.asarray(u[0, :, slice_y]), "--", label="LBM")
    axes[1].set_title(f"Final velocity slice error = {float(rel_l2):.2e}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(f"$u_x(x, y={slice_y})$")
    axes[1].grid(alpha=0.3, linestyle=":")
    axes[1].legend(frameon=False)

    vmax = float(jnp.max(jnp.abs(vorticity)))
    im = axes[2].imshow(
        np.asarray(vorticity.T),
        origin="lower",
        cmap="RdBu_r",
        extent=[0, NX - 1, 0, NY - 1],
        vmin=-vmax,
        vmax=vmax,
    )
    axes[2].quiver(
        grid_x[::quiver_stride, ::quiver_stride],
        grid_y[::quiver_stride, ::quiver_stride],
        np.asarray(u[0, ::quiver_stride, ::quiver_stride].T),
        np.asarray(u[1, ::quiver_stride, ::quiver_stride].T),
        color="black",
        alpha=0.7,
        scale=0.12,
        scale_units="xy",
        angles="xy",
        width=0.004,
    )
    axes[2].set_title("Final vorticity field")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(im, ax=axes[2], label="$\\omega_z$")

    fig.suptitle("2D Taylor-Green vortex decay")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
