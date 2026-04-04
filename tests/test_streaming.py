"""Tests for the LBM streaming step."""

import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest

from vivsim import lbm


class StreamingTest(chex.TestCase):
    """Tests for streaming correctness and conservation."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_shape_preserved(self):
        """Streaming must not change array shape."""
        f = jnp.ones((9, 16, 16))
        f_out = self.variant(lbm.streaming)(f)
        chex.assert_shape(f_out, (9, 16, 16))

    @chex.variants(with_jit=True, without_jit=True)
    def test_mass_conserved(self):
        """Total mass (sum of all populations) must be conserved."""
        key = jax.random.PRNGKey(0)
        f = jax.random.uniform(key, (9, 32, 32), minval=0.01, maxval=0.2)
        f_out = self.variant(lbm.streaming)(f)
        chex.assert_trees_all_close(f.sum(), f_out.sum(), atol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_rest_population_unchanged(self):
        """Direction 0 (rest) has zero velocity, so it should not move."""
        key = jax.random.PRNGKey(1)
        f = jax.random.uniform(key, (9, 16, 16))
        f_out = self.variant(lbm.streaming)(f)
        chex.assert_trees_all_close(f[0], f_out[0], atol=1e-6)

    @chex.variants(with_jit=True, without_jit=True)
    def test_periodic_boundary(self):
        """Streaming wraps around the domain (periodic BC)."""
        f = jnp.zeros((9, 4, 4))
        # Place a value at the right edge for direction 1 (moves +x)
        f = f.at[1, 3, 0].set(1.0)
        f_out = self.variant(lbm.streaming)(f)
        # Should wrap to the left edge
        self.assertAlmostEqual(float(f_out[1, 0, 0]), 1.0, places=6)


if __name__ == "__main__":
    absltest.main()
