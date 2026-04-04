"""Tests for equilibrium distribution and macroscopic quantities."""

import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest

from vivsim import lbm


NX, NY = 16, 16


class EquilibriumTest(chex.TestCase):
    """Tests for the equilibrium distribution function."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_sum_equals_density(self):
        """Equilibrium populations must sum to the density."""
        rho = jnp.ones((NX, NY)) * 1.5
        u = jnp.zeros((2, NX, NY))
        feq = self.variant(lbm.get_equilibrium)(rho, u)
        chex.assert_trees_all_close(jnp.sum(feq, axis=0), rho, atol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_momentum_equals_rho_u(self):
        """First moment of equilibrium must equal rho * u."""
        rho = jnp.ones((NX, NY))
        key = jax.random.PRNGKey(0)
        u = jax.random.uniform(key, (2, NX, NY), minval=-0.1, maxval=0.1)
        feq = self.variant(lbm.get_equilibrium)(rho, u)
        # Recover velocity from feq
        rho_out, u_out = lbm.get_macroscopic(feq)
        chex.assert_trees_all_close(u_out, u, atol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_zero_velocity_isotropic(self):
        """At zero velocity, diagonal populations should be equal by symmetry."""
        rho = jnp.ones((NX, NY)) * 2.0
        u = jnp.zeros((2, NX, NY))
        feq = self.variant(lbm.get_equilibrium)(rho, u)
        # Directions 1,2,3,4 should all be equal
        chex.assert_trees_all_close(feq[1], feq[2], atol=1e-7)
        chex.assert_trees_all_close(feq[1], feq[3], atol=1e-7)
        # Directions 5,6,7,8 should all be equal
        chex.assert_trees_all_close(feq[5], feq[6], atol=1e-7)
        chex.assert_trees_all_close(feq[5], feq[7], atol=1e-7)

    @chex.variants(with_jit=True, without_jit=True)
    def test_shape_1d(self):
        """Equilibrium should work with 1D slices."""
        rho = jnp.ones((NY,))
        u = jnp.zeros((2, NY))
        feq = self.variant(lbm.get_equilibrium)(rho, u)
        chex.assert_shape(feq, (9, NY))


class MacroscopicTest(chex.TestCase):
    """Tests for get_macroscopic."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_roundtrip(self):
        """get_macroscopic(get_equilibrium(rho, u)) should recover rho and u."""
        rho = jnp.ones((NX, NY)) * 1.2
        key = jax.random.PRNGKey(42)
        u = jax.random.uniform(key, (2, NX, NY), minval=-0.05, maxval=0.05)
        feq = lbm.get_equilibrium(rho, u)
        rho_out, u_out = self.variant(lbm.get_macroscopic)(feq)
        chex.assert_trees_all_close(rho_out, rho, atol=1e-5)
        chex.assert_trees_all_close(u_out, u, atol=1e-5)


class OmegaTest(chex.TestCase):
    """Tests for viscosity-omega conversion."""

    def test_known_value(self):
        """nu=1/6 should give omega=1.0 (tau=1)."""
        omega = lbm.get_omega(1.0 / 6.0)
        self.assertAlmostEqual(float(omega), 1.0, places=6)

    def test_stability_bound(self):
        """omega should be less than 2 for any positive viscosity."""
        omega = lbm.get_omega(0.001)
        self.assertLess(float(omega), 2.0)


if __name__ == "__main__":
    absltest.main()
