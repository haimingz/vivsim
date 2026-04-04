"""Tests for LBM collision operators."""

import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest

from vivsim import lbm


NX, NY = 16, 16


def _make_state(key):
    """Create a physically consistent (f, feq) pair."""
    rho = jnp.ones((NX, NY))
    u = jax.random.uniform(key, (2, NX, NY), minval=-0.05, maxval=0.05)
    feq = lbm.get_equilibrium(rho, u)
    # Small perturbation away from equilibrium
    f = feq + jax.random.normal(jax.random.PRNGKey(99), feq.shape) * 1e-4
    return f, feq


class CollisionBGKTest(chex.TestCase):
    """Tests for BGK collision."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_mass_conserved(self):
        """BGK collision must conserve per-cell mass (density)."""
        key = jax.random.PRNGKey(0)
        rho = 1.0 + jax.random.uniform(key, (NX, NY), minval=-0.01, maxval=0.01)
        u = jax.random.uniform(jax.random.PRNGKey(1), (2, NX, NY), minval=-0.05, maxval=0.05)
        feq = lbm.get_equilibrium(rho, u)
        # BGK conserves mass when feq has the same zeroth moment as f
        f_post = self.variant(lbm.collision_bgk)(feq, feq, 1.5)
        chex.assert_trees_all_close(
            jnp.sum(f_post, axis=0), jnp.sum(feq, axis=0), atol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_equilibrium_fixed_point(self):
        """Collision at equilibrium should return equilibrium."""
        rho = jnp.ones((NX, NY))
        u = jnp.zeros((2, NX, NY))
        feq = lbm.get_equilibrium(rho, u)
        omega = 1.0
        f_post = self.variant(lbm.collision_bgk)(feq, feq, omega)
        chex.assert_trees_all_close(f_post, feq, atol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_shape_preserved(self):
        f, feq = _make_state(jax.random.PRNGKey(1))
        f_post = self.variant(lbm.collision_bgk)(f, feq, 1.0)
        chex.assert_shape(f_post, (9, NX, NY))


class CollisionMRTTest(chex.TestCase):
    """Tests for MRT collision."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_mass_conserved(self):
        f, feq = _make_state(jax.random.PRNGKey(2))
        omega = 1.5
        mrt_op = lbm.get_mrt_collision_operator(omega)
        f_post = self.variant(lbm.collision_mrt)(f, feq, mrt_op)
        chex.assert_trees_all_close(
            jnp.sum(f, axis=0), jnp.sum(f_post, axis=0), atol=1e-5
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_equilibrium_fixed_point(self):
        rho = jnp.ones((NX, NY))
        u = jnp.zeros((2, NX, NY))
        feq = lbm.get_equilibrium(rho, u)
        omega = 1.0
        mrt_op = lbm.get_mrt_collision_operator(omega)
        f_post = self.variant(lbm.collision_mrt)(feq, feq, mrt_op)
        chex.assert_trees_all_close(f_post, feq, atol=1e-6)


class CollisionKBCTest(chex.TestCase):
    """Tests for KBC collision."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_shape_preserved(self):
        f, feq = _make_state(jax.random.PRNGKey(3))
        f_post = self.variant(lbm.collision_kbc)(f, feq, 1.5)
        chex.assert_shape(f_post, (9, NX, NY))

    @chex.variants(with_jit=True, without_jit=True)
    def test_equilibrium_fixed_point(self):
        rho = jnp.ones((NX, NY))
        u = jnp.zeros((2, NX, NY))
        feq = lbm.get_equilibrium(rho, u)
        omega = 1.0
        f_post = self.variant(lbm.collision_kbc)(feq, feq, omega)
        chex.assert_trees_all_close(f_post, feq, atol=1e-5)


class CollisionRegularizedTest(chex.TestCase):
    """Tests for regularized BGK collision."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_shape_preserved(self):
        f, feq = _make_state(jax.random.PRNGKey(4))
        f_post = self.variant(lbm.collision_regularized)(f, feq, 1.5)
        chex.assert_shape(f_post, (9, NX, NY))

    @chex.variants(with_jit=True, without_jit=True)
    def test_equilibrium_fixed_point(self):
        rho = jnp.ones((NX, NY))
        u = jnp.zeros((2, NX, NY))
        feq = lbm.get_equilibrium(rho, u)
        omega = 1.0
        f_post = self.variant(lbm.collision_regularized)(feq, feq, omega)
        chex.assert_trees_all_close(f_post, feq, atol=1e-5)


if __name__ == "__main__":
    absltest.main()
