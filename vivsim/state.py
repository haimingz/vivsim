"""Simulation state containers using chex dataclasses.

These dataclasses are automatically registered as JAX pytrees, making them
compatible with ``jax.jit``, ``jax.lax.scan``, ``jax.vmap``, and all other
JAX transformations.  They replace ad-hoc tuples for carrying simulation
state, providing named fields and preventing index-order bugs.
"""

import jax
import chex


@chex.dataclass
class FluidState:
    """State of the lattice Boltzmann fluid.

    Attributes:
        f: Discrete distribution function, shape ``(9, NX, NY)``.
        rho: Macroscopic density, shape ``(NX, NY)``.
        u: Macroscopic velocity, shape ``(2, NX, NY)``.
    """
    f: jax.Array
    rho: jax.Array
    u: jax.Array


@chex.dataclass
class FSIState:
    """State of a fluid-structure interaction simulation.

    Attributes:
        f: Discrete distribution function, shape ``(9, NX, NY)``.
        d: Structural displacement vector.
        v: Structural velocity vector.
        a: Structural acceleration vector.
    """
    f: jax.Array
    d: jax.Array
    v: jax.Array
    a: jax.Array
