"""Bounce-back and specular-reflection boundary conditions for D2Q9 LBM.

These kernels require both the pre-streaming and post-streaming distribution
functions. Unlike the other boundary schemes in this package, they reconstruct
the unknown incoming populations from information that would otherwise be lost
during streaming.
"""

import jax.numpy as jnp

from ..basic import OPP_DIRS
from ._helpers import BOUNDARY_SPEC, get_signed_wall_velocity


def boundary_bounce_back(f_before_stream, f, loc, ux_wall=0, uy_wall=0):
    """Apply no-slip boundary using the bounce-back method on the specified boundary.

    This function should be called after the streaming step.

    In the streaming step, the outgoing distribution functions have been moved to the
    opposite side of the domain due to the `jnp.roll` operation. However, there is a great
    chance that these distribution functions have been modified by other boundary conditions
    before this function is called. Therefore, we need to directly obtain that information 
    from the distribution functions before the streaming step. This can be done by assigning
    the output of `streaming` to a new variable, e.g., `f_new = streaming(f)` for later use 
    and keep the original `f` unchanged.

    In this function, we will reverse the directions of the outgoing distribution functions
    in-place. When the no-slip boundary is moving, we also need to account for the
    momentum exchange due to the moving boundary.

    Args:
        f_before_stream (jax.Array): The DDF before streaming, shape (9, NX, NY).
        f (jax.Array): The DDF after streaming, shape (9, NX, NY).
        loc (str): Boundary location, one of ``'left'``, ``'right'``, ``'top'``,
            or ``'bottom'``.
        ux_wall (scalar or jax.Array of shape N): The x-component of wall velocity.
        uy_wall (scalar or jax.Array of shape N): The y-component of wall velocity.

    Returns:
        jax.Array: Updated distribution function with shape ``(9, NX, NY)``.
    """
    spec = BOUNDARY_SPEC[loc]
    un, ut = get_signed_wall_velocity(loc, ux_wall=ux_wall, uy_wall=uy_wall)

    wall_pre = f_before_stream[:, *spec.wall]  # (9, N) – small slice
    new_vals = jnp.stack([
        wall_pre[spec.out_dirs[0]] + 2 / 3 * un,
        wall_pre[spec.out_dirs[1]] + 1 / 6 * (un + ut),
        wall_pre[spec.out_dirs[2]] + 1 / 6 * (un - ut),
    ])  # (3, N)
    new_wall = f[:, *spec.wall].at[jnp.array(spec.in_dirs)].set(new_vals)
    return f.at[:, *spec.wall].set(new_wall)


def boundary_specular_reflection(f_before_stream, f, loc, ux_wall=0, uy_wall=0):
    """Apply slip boundary using the specular reflection method on the specified boundary.

    This function should be called after the streaming step.

    In the streaming step, the outgoing distribution functions have been moved to the
    opposite side of the domain due to the `jnp.roll` operation. However, there is a great
    chance that these distribution functions have been modified by other boundary conditions
    before this function is called. Therefore, we need to directly obtain that information 
    from the distribution functions before the streaming step. This can be done by assigning
    the output of `streaming` to a new variable, e.g., `f_new = streaming(f)` for later use 
    and keep the original `f` unchanged.

    The difference between specular reflection and bounce-back is that in specular reflection,
    the directions of the hit-wall particles are not reversed, but reflected like a mirror.

    Args:
        f_before_stream (jax.Array): The DDF before streaming, shape ``(9, NX, NY)``.
        f (jax.Array): The DDF after streaming, shape ``(9, NX, NY)``.
        loc (str): Boundary location, one of ``'left'``, ``'right'``, ``'top'``,
            or ``'bottom'``.
        ux_wall (scalar or jax.Array of shape N): The x-component of wall velocity.
        uy_wall (scalar or jax.Array of shape N): The y-component of wall velocity.

    Returns:
        jax.Array: Updated distribution function with shape ``(9, NX, NY)``.
    """
    spec = BOUNDARY_SPEC[loc]
    un, ut = get_signed_wall_velocity(loc, ux_wall=ux_wall, uy_wall=uy_wall)

    wall_pre = f_before_stream[:, *spec.wall]  # (9, N) – small slice
    new_vals = jnp.stack([
        wall_pre[spec.out_dirs[0]] + 2 / 3 * un,
        wall_pre[spec.out_dirs[1]] + 1 / 6 * (un + ut),
        wall_pre[spec.out_dirs[2]] + 1 / 6 * (un - ut),
    ])  # (3, N)
    # specular reflection: diagonal dirs swap (in_dirs[1] ↔ in_dirs[2]) relative to bounce-back
    scatter_dirs = jnp.array([spec.in_dirs[0], spec.in_dirs[2], spec.in_dirs[1]])
    new_wall = f[:, *spec.wall].at[scatter_dirs].set(new_vals)
    return f.at[:, *spec.wall].set(new_wall)


def obstacle_bounce_back(f, mask):
    """Enforce no-slip bounce-back on obstacle cells selected by a boolean mask.

    This helper can also be used on domain boundaries represented as mask cells.

    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        mask (jax.Array of shape (NX, NY)): Boolean mask indicating obstacle cells.

    Returns:
        jax.Array: Updated distribution function with shape ``(9, NX, NY)``.
    """
    return f.at[:, mask].set(f[:, mask][OPP_DIRS])
