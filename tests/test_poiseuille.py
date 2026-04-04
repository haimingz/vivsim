"""End-to-end validation: Poiseuille flow against the analytical solution.

A constant body force drives flow between two no-slip walls.  The steady
state velocity profile is parabolic:

    u_x(y) = (G / 2 nu) * y * (H - y)

where G is the body force, nu the kinematic viscosity, and H the channel
height.  This test runs enough iterations to reach steady state and then
compares the LBM result against the analytical profile.
"""

import jax
import jax.numpy as jnp
from absl.testing import absltest

from vivsim import lbm


# Flow parameters chosen for fast convergence.
NU = 0.2
GX = 1e-3
NX, NY = 10, 10
OMEGA = lbm.get_omega(NU)
N_STEPS = 20_000  # enough for steady state at this grid size


def _analytical_profile(ny, nu, gx):
    """Return the analytical Poiseuille profile for bounce-back walls.

    With NEBB/bounce-back, no-slip walls sit at y = -0.5 and y = NY - 0.5,
    giving an effective channel width of NY - 1 between wall nodes.
    """
    y = jnp.arange(ny)
    return gx / (2 * nu) * y * (ny - 1 - y)


def _run_poiseuille_bgk_edm(n_steps):
    """Run Poiseuille flow with BGK + EDM forcing."""
    rho = jnp.ones((NX, NY))
    u = jnp.zeros((2, NX, NY))
    g = jnp.zeros((2, NX, NY)).at[0].set(GX)
    f = lbm.get_equilibrium(rho, u - g / 2)

    ux_wall, _ = lbm.get_corrected_wall_velocity(0, 0, gx_wall=GX)

    def step(f, _):
        rho_l, u_l = lbm.get_macroscopic(f)
        feq = lbm.get_equilibrium(rho_l, u_l)
        f = lbm.collision_bgk(f, feq, OMEGA)
        f = lbm.forcing_edm(f, g, u_l, rho_l)
        f = lbm.streaming(f)
        f = lbm.boundary_nebb(f, loc="top", ux_wall=ux_wall)
        f = lbm.boundary_nebb(f, loc="bottom", ux_wall=ux_wall)
        return f, None

    f, _ = jax.lax.scan(step, f, None, length=n_steps)

    rho_f, u_f = lbm.get_macroscopic(f)
    u_f = u_f + lbm.get_velocity_correction(g, rho_f)
    return u_f


def _run_poiseuille_bgk_guo(n_steps):
    """Run Poiseuille flow with BGK + Guo forcing."""
    rho = jnp.ones((NX, NY))
    u = jnp.zeros((2, NX, NY))
    g = jnp.zeros((2, NX, NY)).at[0].set(GX)
    f = lbm.get_equilibrium(rho, u - g / 2)

    ux_wall, _ = lbm.get_corrected_wall_velocity(0, 0, gx_wall=GX)

    def step(f, _):
        rho_l, u_l = lbm.get_macroscopic(f)
        u_l = u_l + lbm.get_velocity_correction(g, rho_l)
        feq = lbm.get_equilibrium(rho_l, u_l)
        f = lbm.collision_bgk(f, feq, OMEGA)
        f = lbm.forcing_guo_bgk(f, g, u_l, OMEGA)
        f = lbm.streaming(f)
        f = lbm.boundary_nebb(f, loc="top", ux_wall=ux_wall)
        f = lbm.boundary_nebb(f, loc="bottom", ux_wall=ux_wall)
        return f, None

    f, _ = jax.lax.scan(step, f, None, length=n_steps)

    rho_f, u_f = lbm.get_macroscopic(f)
    u_f = u_f + lbm.get_velocity_correction(g, rho_f)
    return u_f


class PoiseuilleBGKEDMTest(absltest.TestCase):
    """Validate BGK + EDM against the analytical Poiseuille solution."""

    def test_profile(self):
        u = _run_poiseuille_bgk_edm(N_STEPS)
        ux_mid = u[0, NX // 2, :]
        ux_analytical = _analytical_profile(NY, NU, GX)
        max_err = float(jnp.max(jnp.abs(ux_mid - ux_analytical)))
        # Relative error w.r.t. peak velocity
        peak = float(jnp.max(ux_analytical))
        rel_err = max_err / peak
        self.assertLess(rel_err, 0.02, f"Relative error {rel_err:.4f} exceeds 2%")


class PoiseuilleBGKGuoTest(absltest.TestCase):
    """Validate BGK + Guo forcing against the analytical Poiseuille solution."""

    def test_profile(self):
        u = _run_poiseuille_bgk_guo(N_STEPS)
        ux_mid = u[0, NX // 2, :]
        ux_analytical = _analytical_profile(NY, NU, GX)
        max_err = float(jnp.max(jnp.abs(ux_mid - ux_analytical)))
        peak = float(jnp.max(ux_analytical))
        rel_err = max_err / peak
        self.assertLess(rel_err, 0.02, f"Relative error {rel_err:.4f} exceeds 2%")


if __name__ == "__main__":
    absltest.main()
