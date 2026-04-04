"""
Poiseuille flow in a 2D channel driven by a constant body force using LBM

This example is performed using different LBM collision models and forcing schemes.
The results are compared with the analytical solution to validate the implementation.
"""


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from vivsim import lbm


NU = 0.2 # kinematic viscosity
GX = 0.001  # body force in x direction

NX, NY = 10, 10  # number of lattice nodes in x and y directions
OMEGA = lbm.get_omega(NU)  # relaxation parameter
TM = 100000  # number of time steps


# initialize the macroscopic variables
rho = jnp.ones((NX, NY))
u = jnp.zeros((2, NX, NY))

g = jnp.zeros((2, NX, NY)).at[0].set(GX)

# initialize the distribution function to equilibrium state
f = lbm.get_equilibrium(rho, u - g / 2)


# In the following, we provide several simulation recipes using different
# combinations of collision models and forcing schemes.

# Note that the velocity correction is and only is necessary when external forces are present,
# This correction is due to the redefinition of the distribution function for second-order accuracy.
# Thus, it is independent of the collision operator and forcing scheme used.

# Since the external force is also present at the wall, the velocity correction should also be
# considered in the boundary condition step. Pre-compute the corrected (no-slip) wall velocity
# once and pass it to every boundary call.
UX_WALL, _ = lbm.get_corrected_wall_velocity(0, 0, gx_wall=GX)

MRT_COLLISION = lbm.get_mrt_collision_operator(OMEGA)
MRT_FORCING = lbm.get_mrt_forcing_operator(OMEGA)


@jax.jit
def main_bgk_edm(f):
    """BGK collision + Exact Difference Method forcing.

    EDM requires uncorrected velocity for collision & forcing,
    so the velocity correction is applied AFTER.
    """
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_bgk(f, feq, OMEGA)
    f = lbm.forcing_edm(f, g, u, rho)
    u = u + lbm.get_velocity_correction(g, rho)
    f = lbm.streaming(f)
    f = lbm.boundary_nebb(f, loc='top', ux_wall=UX_WALL)
    f = lbm.boundary_nebb(f, loc='bottom', ux_wall=UX_WALL)
    return f, rho, u


@jax.jit
def main_bgk_guo(f):
    """BGK collision + Guo forcing.

    Guo requires corrected velocity for collision & forcing,
    so the velocity correction is applied BEFORE.
    """
    rho, u = lbm.get_macroscopic(f)
    u = u + lbm.get_velocity_correction(g, rho)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_bgk(f, feq, OMEGA)
    f = lbm.forcing_guo_bgk(f, g, u, OMEGA)
    f = lbm.streaming(f)
    f = lbm.boundary_nebb(f, loc='top', ux_wall=UX_WALL)
    f = lbm.boundary_nebb(f, loc='bottom', ux_wall=UX_WALL)
    return f, rho, u


@jax.jit
def main_mrt_guo(f):
    """MRT collision + Guo forcing.

    Guo requires corrected velocity for collision & forcing,
    so the velocity correction is applied BEFORE.
    (This is not as accurate as the other recipes.)
    """
    rho, u = lbm.get_macroscopic(f)
    u = u + lbm.get_velocity_correction(g, rho)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_mrt(f, feq, MRT_COLLISION)
    f = lbm.forcing_guo_mrt(f, g, u, MRT_FORCING)
    f = lbm.streaming(f)
    f = lbm.boundary_nebb(f, loc='top', ux_wall=UX_WALL)
    f = lbm.boundary_nebb(f, loc='bottom', ux_wall=UX_WALL)
    return f, rho, u


@jax.jit
def main_kbc_edm(f):
    """KBC collision + Exact Difference Method forcing.

    EDM requires uncorrected velocity for collision & forcing,
    so the velocity correction is applied AFTER.
    """
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)
    f = lbm.forcing_edm(f, g, u, rho)
    u = u + lbm.get_velocity_correction(g, rho)
    f = lbm.streaming(f)
    f = lbm.boundary_nebb(f, loc='top', ux_wall=UX_WALL)
    f = lbm.boundary_nebb(f, loc='bottom', ux_wall=UX_WALL)
    return f, rho, u


# ======================= SELECT RECIPE AND RUN ==========================

RECIPES = {
    "bgk_edm": main_bgk_edm,
    "bgk_guo": main_bgk_guo,
    "mrt_guo": main_mrt_guo,
    "kbc_edm": main_kbc_edm,
}

recipe = RECIPES["kbc_edm"]  # <-- change this to select a different recipe

def run(f, n_steps):
    def step(f, _):
        f, _, _ = recipe(f)
        return f, None
    f, _ = jax.lax.scan(step, f, None, length=n_steps)
    # Final step to get rho, u
    _, rho, u = recipe(f)
    return f, rho, u

run = jax.jit(run, static_argnums=1)

f, rho, u = run(f, TM)

# ======================= COMPARE WITH ANALYTICAL SOLUTION ================

H = NY - 1
y = jnp.arange(0, H, 0.01)
ux_true = GX / 2 / NU * (y * (H - y))

plt.plot(y, ux_true, '-', label='Analytical')
plt.plot(jnp.arange(NY), u[0, NX//2], '+', label='LBM')
plt.legend()
plt.xlabel('$y$')
plt.ylabel('$u_x$')
plt.title('Poiseuille flow in a channel')
plt.show()
