"""Tests for chex dataclass state containers."""

import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest

from vivsim.state import FluidState, FSIState


class FluidStateTest(chex.TestCase):
    """Tests for FluidState pytree compatibility."""

    def test_jit_compatible(self):
        """FluidState should work with jax.jit."""
        state = FluidState(
            f=jnp.ones((9, 16, 16)),
            rho=jnp.ones((16, 16)),
            u=jnp.zeros((2, 16, 16)),
        )

        @jax.jit
        def identity(s):
            return s

        out = identity(state)
        chex.assert_trees_all_close(out.f, state.f)

    def test_tree_map(self):
        """FluidState should work with jax.tree.map."""
        state = FluidState(
            f=jnp.ones((9, 4, 4)),
            rho=jnp.ones((4, 4)),
            u=jnp.zeros((2, 4, 4)),
        )
        doubled = jax.tree.map(lambda x: x * 2, state)
        chex.assert_trees_all_close(doubled.f, jnp.ones((9, 4, 4)) * 2)


class FSIStateTest(chex.TestCase):
    """Tests for FSIState pytree compatibility."""

    def test_scan_compatible(self):
        """FSIState should work as carry in jax.lax.scan."""
        state = FSIState(
            f=jnp.ones((9, 8, 8)),
            d=jnp.zeros(2),
            v=jnp.zeros(2),
            a=jnp.zeros(2),
        )

        def step(carry, _):
            return carry, None

        final, _ = jax.lax.scan(step, state, None, length=3)
        chex.assert_trees_all_close(final.f, state.f)


if __name__ == "__main__":
    absltest.main()
