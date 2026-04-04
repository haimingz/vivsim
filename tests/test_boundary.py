"""Tests for LBM boundary conditions."""

from functools import partial

import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest

from vivsim import lbm


NX, NY = 32, 32


def _make_equilibrium_field(ux=0.0, uy=0.0):
    """Create an equilibrium state with uniform velocity."""
    rho = jnp.ones((NX, NY))
    u = jnp.zeros((2, NX, NY))
    u = u.at[0].set(ux)
    u = u.at[1].set(uy)
    f = lbm.get_equilibrium(rho, u)
    return f


class BoundaryNEETest(chex.TestCase):
    """Tests for Non-Equilibrium Extrapolation boundaries.

    Boundary functions take a string ``loc`` argument which is traced as a
    static value. We test both the eager path and a JIT path where ``loc``
    is bound via ``functools.partial``.
    """

    def test_shape_preserved(self):
        f = _make_equilibrium_field()
        for loc in ["left", "right", "top", "bottom"]:
            f_out = lbm.boundary_nee(f, loc=loc)
            chex.assert_shape(f_out, (9, NX, NY))

    def test_shape_preserved_jit(self):
        f = _make_equilibrium_field()
        for loc in ["left", "right", "top", "bottom"]:
            f_out = jax.jit(partial(lbm.boundary_nee, loc=loc))(f)
            chex.assert_shape(f_out, (9, NX, NY))

    def test_no_change_at_equilibrium(self):
        """At uniform equilibrium with matching wall values, BC is a no-op."""
        f = _make_equilibrium_field()
        f_streamed = lbm.streaming(f)
        for loc in ["left", "right", "top", "bottom"]:
            f_out = lbm.boundary_nee(f_streamed, loc=loc)
            chex.assert_trees_all_close(f_out, f_streamed, atol=1e-5)


class BoundaryNEBBTest(chex.TestCase):
    """Tests for Non-Equilibrium Bounce-Back (Zou/He) boundaries."""

    def test_shape_preserved(self):
        f = _make_equilibrium_field()
        for loc in ["left", "right", "top", "bottom"]:
            f_out = lbm.boundary_nebb(f, loc=loc)
            chex.assert_shape(f_out, (9, NX, NY))

    def test_shape_preserved_jit(self):
        f = _make_equilibrium_field()
        for loc in ["left", "right", "top", "bottom"]:
            f_out = jax.jit(partial(lbm.boundary_nebb, loc=loc))(f)
            chex.assert_shape(f_out, (9, NX, NY))


class BoundaryBounceBackTest(chex.TestCase):
    """Tests for bounce-back boundaries."""

    def test_shape_preserved(self):
        f = _make_equilibrium_field()
        f_streamed = lbm.streaming(f)
        for loc in ["left", "right", "top", "bottom"]:
            f_out = lbm.boundary_bounce_back(f, f_streamed, loc=loc)
            chex.assert_shape(f_out, (9, NX, NY))

    def test_shape_preserved_jit(self):
        f = _make_equilibrium_field()
        f_streamed = lbm.streaming(f)
        for loc in ["left", "right", "top", "bottom"]:
            fn = jax.jit(partial(lbm.boundary_bounce_back, loc=loc))
            f_out = fn(f, f_streamed)
            chex.assert_shape(f_out, (9, NX, NY))


class ObstacleBounceBackTest(chex.TestCase):
    """Tests for obstacle bounce-back (now JIT-compatible via jnp.where)."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_symmetric_obstacle(self):
        """A symmetric obstacle with zero velocity should preserve symmetry."""
        f = _make_equilibrium_field()
        mask = jnp.zeros((NX, NY), dtype=bool)
        mask = mask.at[NX // 2, NY // 2].set(True)
        f_out = self.variant(lbm.obstacle_bounce_back)(f, mask)
        chex.assert_shape(f_out, (9, NX, NY))


if __name__ == "__main__":
    absltest.main()
