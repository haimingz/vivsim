r"""
3D Arnold-Beltrami-Childress (ABC) flow decay using D3Q19 LBM.

The ABC flow is a classic fully three-dimensional, divergence-free Beltrami
velocity field on a periodic cube. Its vortex lines form intertwined tubes,
making it a compact visual example for 3D periodic LBM solvers.

"""

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from tqdm import tqdm

from vivsim import lbm3d, post


# ========================== SIMULATION PARAMETERS ====================

RENDER_3D = True
CHUNK_STEPS = 100
SAMPLE_EVERY = 100


# ========================== GEOMETRY =======================

NX = 64
NY = 64
NZ = 64

K = 2 * jnp.pi / NX
K2 = K ** 2

X = jnp.arange(NX, dtype=jnp.float32)[:, None, None]
Y = jnp.arange(NY, dtype=jnp.float32)[None, :, None]
Z = jnp.arange(NZ, dtype=jnp.float32)[None, None, :]


# ========================== PHYSICAL PARAMETERS =====================

A = 1.0
B = 1.0
C = 1.0

U0 = 0.02
NU = 0.005
RE = U0 * NX / (2 * jnp.pi) / NU

OMEGA = lbm3d.get_omega(NU)
TM = 2000


# ======================= HELPER FUNCTIONS ====================

def exact_velocity(t):
    """Compute the analytical ABC velocity field at time t."""
    decay = jnp.exp(-NU * K2 * t)
    ux = U0 * (A * jnp.sin(K * Z) + C * jnp.cos(K * Y)) * decay
    uy = U0 * (B * jnp.sin(K * X) + A * jnp.cos(K * Z)) * decay
    uz = U0 * (C * jnp.sin(K * Y) + B * jnp.cos(K * X)) * decay
    ux = jnp.broadcast_to(ux, (NX, NY, NZ))
    uy = jnp.broadcast_to(uy, (NX, NY, NZ))
    uz = jnp.broadcast_to(uz, (NX, NY, NZ))
    return jnp.stack((ux, uy, uz))


def kinetic_energy(u):
    """Compute the mean kinetic energy of the 3D velocity field."""
    return 0.5 * jnp.mean(jnp.sum(u ** 2, axis=0))





# ======================= INITIALIZE VARIABLES ====================

rho = jnp.ones((NX, NY, NZ))
u = exact_velocity(0.0)
f = lbm3d.get_equilibrium(rho, u)


# ======================= SIMULATION ROUTINE =====================

def update_step(f):
    rho, u = lbm3d.get_macroscopic(f)
    feq = lbm3d.get_equilibrium(rho, u)
    f = lbm3d.collision_reg(f, feq, OMEGA)
    f = lbm3d.streaming(f)
    return f


@partial(jax.jit, static_argnums=1, donate_argnums=0)
def update_chunk(carry, n_steps):
    def step(carry, _):
        (f,) = carry
        f = update_step(f)
        return (f,), None
    return jax.lax.scan(step, carry, None, length=n_steps)


# ========================== RUN SIMULATION ==========================

energy_hist = [float(kinetic_energy(u))]

chunk_sizes = [CHUNK_STEPS] * (TM // CHUNK_STEPS)
if TM % CHUNK_STEPS:
    chunk_sizes.append(TM % CHUNK_STEPS)

current_step = 0
with tqdm(total=TM, unit="step") as pbar:
    for n_steps in chunk_sizes:
        (f,), _ = update_chunk((f,), n_steps)
        jax.block_until_ready(f)
        current_step += n_steps
        pbar.update(n_steps)

        if current_step % SAMPLE_EVERY == 0:
            _, u_now = lbm3d.get_macroscopic(f)
            energy_hist.append(float(kinetic_energy(u_now)))


# ========================= POST-PROCESSING ==========================

_, u = lbm3d.get_macroscopic(f)
u_true = exact_velocity(float(TM))
rel_l2 = float(jnp.linalg.norm(u - u_true) / jnp.linalg.norm(u_true))

q_crit = np.asarray(post.q_criterion(u))
vorticity = np.asarray(post.vorticity(u))
vort_mag = np.asarray(post.vorticity_magnitude(u))

sample_steps_arr = np.arange(0, TM + 1, SAMPLE_EVERY)
energy_hist_arr = np.array(energy_hist[:len(sample_steps_arr)])
energy_true = energy_hist_arr[0] * np.exp(-2 * float(NU * K2) * sample_steps_arr)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(sample_steps_arr, energy_true, label="Analytical", linewidth=2)
ax.plot(sample_steps_arr, energy_hist_arr, "o", ms=3, label="LBM")
ax.set_title("ABC flow: kinetic energy decay")
ax.set_xlabel("Time step")
ax.set_ylabel("Kinetic energy")
ax.grid(alpha=0.3, linestyle=":")
ax.legend(frameon=False)
fig.tight_layout()

slice_z = NZ // 4
omega_z = vorticity[2, :, :, slice_z]
omega_lim = max(float(np.percentile(np.abs(omega_z), 99.5)), 1e-12)

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(
    omega_z.T,
    origin="lower",
    cmap="RdBu_r",
    extent=[0, NX - 1, 0, NY - 1],
    vmin=-omega_lim,
    vmax=omega_lim,
)
ax.set_title(f"Signed z-vorticity at z = {slice_z}")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.colorbar(im, ax=ax, label="$\\omega_z$")
fig.tight_layout()

if RENDER_3D:
    grid = pv.ImageData(dimensions=(NX + 1, NY + 1, NZ + 1))
    grid.cell_data["Q-criterion"] = q_crit.ravel(order="F")
    grid.cell_data["vorticity_magnitude"] = vort_mag.ravel(order="F")

    positive_q = q_crit[q_crit > 0]
    point_grid = grid.cell_data_to_point_data()
    iso = None
    if positive_q.size:
        q_threshold = float(np.percentile(positive_q, 80))
        iso = point_grid.contour(
            isosurfaces=[q_threshold],
            scalars="Q-criterion",
        )

    pl = pv.Plotter()
    pl.set_background("white")
    pl.camera_position = [
        (2.45 * NX, 2.0 * NY, 1.75 * NZ),
        (0.5 * NX, 0.5 * NY, 0.5 * NZ),
        (0.0, 0.0, 1.0),
    ]
    pl.add_mesh(
        pv.Box(bounds=(0, NX - 1, 0, NY - 1, 0, NZ - 1)),
        style="wireframe",
        color="black",
        opacity=0.18,
        line_width=1.0,
    )
    if iso is not None and iso.n_points > 0:
        iso["vorticity_magnitude"] = iso.sample(point_grid)["vorticity_magnitude"]
        pl.add_mesh(
            iso,
            scalars="vorticity_magnitude",
            cmap="plasma",
            opacity=0.82,
            smooth_shading=True,
            show_scalar_bar=True,
            scalar_bar_args={"title": "|omega|"},
        )
    pl.add_text(
        f"ABC flow  (Re ~ {float(RE):.0f}, t = {TM}, error = {rel_l2:.2e})",
        position="upper_left",
        font_size=10,
        color="black",
    )
    pl.show_axes()
    pl.reset_camera_clipping_range()
    pl.show()

plt.show()
