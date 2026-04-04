"""Tests for the Immersed Boundary module."""

import chex
import jax.numpy as jnp
from absl.testing import absltest

from vivsim import ib


class StencilTest(chex.TestCase):
    """Tests for IB stencil computation."""

    def test_stencil_shape(self):
        """Stencil weights and indices have shape (n_markers, n_stencil)."""
        n_markers = 10
        ny = 32
        marker_x = jnp.linspace(8.0, 24.0, n_markers)
        marker_y = jnp.linspace(8.0, 24.0, n_markers)
        weights, indices = ib.get_ib_stencil(marker_x, marker_y, ny)
        n_stencil = 4 * 4  # stencil_radius=2 -> 4x4 stencil
        chex.assert_shape(weights, (n_markers, n_stencil))
        chex.assert_shape(indices, (n_markers, n_stencil))

    def test_weights_positive(self):
        """All stencil weights should be non-negative for Peskin kernels."""
        marker_x = jnp.array([10.5, 15.3])
        marker_y = jnp.array([10.5, 15.7])
        weights, _ = ib.get_ib_stencil(marker_x, marker_y, ny=32)
        self.assertTrue(jnp.all(weights >= 0))


class InterpolateSpreadTest(chex.TestCase):
    """Tests for interpolation and spreading."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_interpolate_uniform_field(self):
        """Interpolating a uniform field should return the uniform value."""
        nx, ny = 32, 32
        n_markers = 5
        grid_values = jnp.ones((2, nx, ny)) * 3.0
        marker_x = jnp.linspace(5.0, 25.0, n_markers)
        marker_y = jnp.linspace(5.0, 25.0, n_markers)
        weights, indices = ib.get_ib_stencil(marker_x, marker_y, ny)
        result = self.variant(ib.interpolate)(grid_values, weights, indices)
        chex.assert_shape(result, (n_markers, 2))
        chex.assert_trees_all_close(result, jnp.ones((n_markers, 2)) * 3.0, atol=1e-5)

    @chex.variants(with_jit=True, without_jit=True)
    def test_spread_shape(self):
        """Spread should return same shape as the grid."""
        nx, ny = 32, 32
        n_markers = 5
        marker_values = jnp.ones((n_markers, 2))
        grid_values = jnp.zeros((2, nx, ny))
        marker_x = jnp.linspace(5.0, 25.0, n_markers)
        marker_y = jnp.linspace(5.0, 25.0, n_markers)
        weights, indices = ib.get_ib_stencil(marker_x, marker_y, ny)
        result = self.variant(ib.spread)(marker_values, grid_values, weights, indices)
        chex.assert_shape(result, (2, nx, ny))


class GeometryTest(chex.TestCase):
    """Tests for IB geometry functions."""

    def test_unit_square_area(self):
        """Area of a unit square should be 1.0."""
        coords = jnp.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=jnp.float32)
        area = ib.get_area(coords)
        self.assertAlmostEqual(float(area), 1.0, places=5)

    def test_segment_lengths_closed(self):
        """Segment lengths of a unit square should all be 1.0."""
        coords = jnp.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=jnp.float32)
        ds = ib.get_ds_closed(coords)
        chex.assert_shape(ds, (4,))
        chex.assert_trees_all_close(ds, jnp.ones(4), atol=1e-5)


if __name__ == "__main__":
    absltest.main()
