"""Tests for LBM forcing schemes (Guo and EDM)."""

import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest

from vivsim import lbm


NX, NY = 16, 16


class GuoForcingTermTest(chex.TestCase):
    """Tests for the Guo lattice forcing term."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_shape(self):
        """Forcing term should have shape (9, NX, NY)."""
        g = jnp.zeros((2, NX, NY)).at[0].set(1e-5)
        u = jnp.zeros((2, NX, NY))
        g_lat = self.variant(lbm.get_guo_forcing_term)(g, u)
        chex.assert_shape(g_lat, (9, NX, NY))

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero_force_zero_term(self):
        """Zero external force should produce zero forcing term."""
        g = jnp.zeros((2, NX, NY))
        u = jnp.zeros((2, NX, NY))
        g_lat = self.variant(lbm.get_guo_forcing_term)(g, u)
        chex.assert_trees_all_close(g_lat, jnp.zeros_like(g_lat), atol=1e-7)

    @chex.variants(with_jit=True, without_jit=True)
    def test_sum_equals_zero(self):
        """Sum over directions of the Guo forcing term should be zero.

        This ensures the forcing does not create or destroy mass.
        """
        g = jnp.zeros((2, NX, NY)).at[0].set(1e-4)
        u = jnp.zeros((2, NX, NY))
        g_lat = self.variant(lbm.get_guo_forcing_term)(g, u)
        chex.assert_trees_all_close(
            jnp.sum(g_lat, axis=0), jnp.zeros((NX, NY)), atol=1e-7
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_first_moment_equals_force(self):
        """First moment of forcing term should recover the applied force.

        sum_i g_lat_i * c_i = g  (at zero velocity).
        """
        from vivsim.lbm.basic import VELOCITIES

        g = jnp.zeros((2, NX, NY)).at[0].set(1e-4)
        u = jnp.zeros((2, NX, NY))
        g_lat = self.variant(lbm.get_guo_forcing_term)(g, u)
        # First moment: sum_i g_lat_i * c_ia
        momentum = jnp.einsum("ixy,ia->axy", g_lat, VELOCITIES)
        chex.assert_trees_all_close(momentum, g, atol=1e-7)


class ForcingGuoBGKTest(chex.TestCase):
    """Tests for Guo forcing applied to BGK."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_shape(self):
        rho = jnp.ones((NX, NY))
        u = jnp.zeros((2, NX, NY))
        f = lbm.get_equilibrium(rho, u)
        g = jnp.zeros((2, NX, NY)).at[0].set(1e-5)
        omega = 1.0
        f_out = self.variant(lbm.forcing_guo_bgk)(f, g, u, omega)
        chex.assert_shape(f_out, (9, NX, NY))


class ForcingGuoMRTTest(chex.TestCase):
    """Tests for Guo forcing applied to MRT."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_shape(self):
        rho = jnp.ones((NX, NY))
        u = jnp.zeros((2, NX, NY))
        f = lbm.get_equilibrium(rho, u)
        g = jnp.zeros((2, NX, NY)).at[0].set(1e-5)
        omega = 1.0
        mrt_fop = lbm.get_mrt_forcing_operator(omega)
        f_out = self.variant(lbm.forcing_guo_mrt)(f, g, u, mrt_fop)
        chex.assert_shape(f_out, (9, NX, NY))


class ForcingEDMTest(chex.TestCase):
    """Tests for Exact Difference Method forcing."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_shape(self):
        rho = jnp.ones((NX, NY))
        u = jnp.zeros((2, NX, NY))
        f = lbm.get_equilibrium(rho, u)
        g = jnp.zeros((2, NX, NY)).at[0].set(1e-5)
        f_out = self.variant(lbm.forcing_edm)(f, g, u, rho)
        chex.assert_shape(f_out, (9, NX, NY))

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero_force_no_change(self):
        """Zero force should not alter the distribution function."""
        rho = jnp.ones((NX, NY))
        u = jnp.zeros((2, NX, NY))
        f = lbm.get_equilibrium(rho, u)
        g = jnp.zeros((2, NX, NY))
        f_out = self.variant(lbm.forcing_edm)(f, g, u, rho)
        chex.assert_trees_all_close(f_out, f, atol=1e-7)

    @chex.variants(with_jit=True, without_jit=True)
    def test_mass_conserved(self):
        """EDM should conserve total mass (sum over directions unchanged)."""
        rho = jnp.ones((NX, NY))
        u = jnp.zeros((2, NX, NY))
        f = lbm.get_equilibrium(rho, u)
        g = jnp.zeros((2, NX, NY)).at[0].set(1e-4)
        f_out = self.variant(lbm.forcing_edm)(f, g, u, rho)
        chex.assert_trees_all_close(
            jnp.sum(f_out, axis=0), jnp.sum(f, axis=0), atol=1e-6
        )


if __name__ == "__main__":
    absltest.main()
