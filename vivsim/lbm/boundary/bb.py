"""
This module implements boundary conditions using the bounce-back and specular reflection methods
for the Lattice Boltzmann Method (LBM) in a 2D D2Q9 lattice.

Note that when using these boundary conditions, the distribution functions before streaming, i.e.,
``f_before_stream``, must also be provided. This is different from other boundary conditions that work well
with only the post-streaming distribution functions. This requirement arises because the bounce-back
and specular reflection methods require the distribution functions that are going outside the domain
in the streaming step. Thus, a pre-streaming distribution function is needed so that we can access the
preserved information of the outgoing distribution functions.
"""

import jax.numpy as jnp
from ..basic import OPP_DIRS
from ._slices import SPATIAL_WALL_SLICE, NEBB_CONFIG, validate_loc


def boundary_bounce_back(f_before_stream, f, loc, ux_wall=0, uy_wall=0):
    """Apply no-slip boundary using the bounce-back method on the specified boundary.

    This function should be called after the streaming step.

    In the streaming step, the outgoing distribution functions have been moved to the
    opposite side of the domain due to the ``jnp.roll`` operation. However, there is a great
    chance that these distribution functions have been modified by other boundary conditions
    before this function is called. Therefore, we need to directly obtain that information
    from the distribution functions before the streaming step. This can be done by assigning
    the output of ``streaming`` to a new variable, e.g., ``f_new = streaming(f)`` for later use
    and keep the original ``f`` unchanged.

    In this function, we will reverse the directions of the outgoing distribution functions
    in-place. When the no-slip boundary is moving, we also need to account for the
    momentum exchange due to the moving boundary.

    Args:
        f_before_stream (jax.Array): The DDF before streaming, shape (9, NX, NY).
        f (jax.Array): The DDF after streaming, shape (9, NX, NY).
        loc (str): The location of the boundary where the bounce-back is applied.
                   Should be one of 'left', 'right', 'top', or 'bottom'.
        ux_wall (scalar or jax.Array of shape NX): The x-component of the wall velocity.
        uy_wall (scalar or jax.Array of shape NY): The y-component of the wall velocity.

    Returns:
        f (jax.Array): The DDF after applying the bounce-back boundary condition, shape (9, NX, NY).
    """
    validate_loc(loc)
    cfg = NEBB_CONFIG[loc]
    sws = SPATIAL_WALL_SLICE[loc]
    i0, i1, i2 = cfg["incoming"]
    s0, s1, s2 = cfg["source"]
    sn = cfg["normal_sign"]
    un = ux_wall if cfg["normal_axis"] == 0 else uy_wall
    ut = uy_wall if cfg["normal_axis"] == 0 else ux_wall

    f = f.at[(i0,) + sws].set(f_before_stream[(s0,) + sws] + sn * 2 / 3 * un)
    f = f.at[(i1,) + sws].set(f_before_stream[(s1,) + sws] + sn * 1 / 6 * (un + ut))
    f = f.at[(i2,) + sws].set(f_before_stream[(s2,) + sws] + sn * 1 / 6 * (un - ut))
    return f


def boundary_specular_reflection(f_before_stream, f, loc, ux_wall=0, uy_wall=0):
    """Apply slip boundary using the specular reflection method on the specified boundary.

    This function should be called after the streaming step.

    The difference between specular reflection and bounce-back is that in specular reflection,
    the directions of the hit-wall particles are not reversed, but reflected like a mirror.
    The diagonal incoming directions swap their sources compared to bounce-back.

    Args:
        f_before_stream (jax.Array): The DDF before streaming, shape (9, NX, NY).
        f (jax.Array): The DDF after streaming, shape (9, NX, NY).
        loc (str): The location of the boundary where the specular reflection is applied.
                   Should be one of 'left', 'right', 'top', or 'bottom'.
        ux_wall (scalar or jax.Array of shape NX): The x-component of the wall velocity.
        uy_wall (scalar or jax.Array of shape NY): The y-component of the wall velocity.

    Returns:
        f (jax.Array): The DDF after applying the specular reflection condition, shape (9, NX, NY).
    """
    validate_loc(loc)
    cfg = NEBB_CONFIG[loc]
    sws = SPATIAL_WALL_SLICE[loc]
    i0, i1, i2 = cfg["incoming"]
    s0, s1, s2 = cfg["source"]
    sn = cfg["normal_sign"]
    un = ux_wall if cfg["normal_axis"] == 0 else uy_wall
    ut = uy_wall if cfg["normal_axis"] == 0 else ux_wall

    # Normal direction: same as bounce-back
    f = f.at[(i0,) + sws].set(f_before_stream[(s0,) + sws] + sn * 2 / 3 * un)
    # Diagonal directions: sources swapped compared to bounce-back (mirror reflection)
    f = f.at[(i2,) + sws].set(f_before_stream[(s1,) + sws] + sn * 1 / 6 * (un + ut))
    f = f.at[(i1,) + sws].set(f_before_stream[(s2,) + sws] + sn * 1 / 6 * (un - ut))
    return f


def obstacle_bounce_back(f, mask):
    """Enforce a no-slip boundary using the Bounce Back scheme
    at any obstacles defined by a 2D mask, where True indicates
    the presence of an obstacle.

    This can also be used to enforce no-slip boundaries at domain boundaries.

    Uses ``jnp.where`` instead of boolean fancy-indexing so that array shapes
    are static, enabling full JIT compilation on all backends.

    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        mask (jax.Array of shape (NX, NY)): The mask indicating the obstacle.

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """
    f_reversed = f[OPP_DIRS]
    return jnp.where(mask[jnp.newaxis, :, :], f_reversed, f)
