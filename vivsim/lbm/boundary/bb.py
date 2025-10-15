"""
This module implements boundary conditions using the bounce-back and specular reflection methods
for the Lattice Boltzmann Method (LBM) in a 2D D2Q9 lattice. 

Note that when using these boundary conditions, the distribution functions before streaming, i.e.,
`f_before_stream`, must also be provided. This is different from other boundary conditions that work well
with only the post-streaming distribution functions. This requirement arises because the bounce-back
and specular reflection methods require the distribution functions that are going outside the domain
in the streaming step. Thus, a pre-streaming distribution function is needed so that we can access the
preserved information of the outgoing distribution functions.
"""

from ..basic import OPP_DIRS


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
        loc (str): The location of the boundary where the bounce-back is applied.
                   Should be one of 'left', 'right', 'top', or 'bottom'.
        ux_wall (scalar or jax.Array of shape NX): The x-component of the wall velocity.
        uy_wall (scalar or jax.Array of shape NY): The y-component of the wall velocity.

    Returns:
        f (jax.Array): The DDF after applying the bounce-back boundary condition, shape (9, NX, NY).
    """

    if loc not in ["left", "right", "top", "bottom"]:
        raise ValueError(
            "Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'."
        )

    if loc == "left":
        f = f.at[1, 0].set(f_before_stream[3, 0] + 2 / 3 * ux_wall)
        f = f.at[5, 0].set(f_before_stream[7, 0] - 1 / 6 * (-ux_wall - uy_wall))
        f = f.at[8, 0].set(f_before_stream[6, 0] - 1 / 6 * (-ux_wall + uy_wall))

    elif loc == "right":
        f = f.at[3, -1].set(f_before_stream[1, -1] - 2 / 3 * ux_wall)
        f = f.at[7, -1].set(f_before_stream[5, -1] - 1 / 6 * (ux_wall + uy_wall))
        f = f.at[6, -1].set(f_before_stream[8, -1] - 1 / 6 * (ux_wall - uy_wall))

    elif loc == "top":
        f = f.at[4, :, -1].set(f_before_stream[2, :, -1] - 2 / 3 * uy_wall)
        f = f.at[7, :, -1].set(f_before_stream[5, :, -1] - 1 / 6 * (ux_wall + uy_wall))
        f = f.at[8, :, -1].set(f_before_stream[6, :, -1] - 1 / 6 * (-ux_wall + uy_wall))

    elif loc == "bottom":
        f = f.at[2, :, 0].set(f_before_stream[4, :, 0] + 2 / 3 * uy_wall)
        f = f.at[5, :, 0].set(f_before_stream[7, :, 0] - 1 / 6 * (-ux_wall - uy_wall))
        f = f.at[6, :, 0].set(f_before_stream[8, :, 0] - 1 / 6 * (ux_wall - uy_wall))

    return f


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
        f (jax.Array): The DDF after streaming, shape (9, NX, NY).
        f_before_stream (jax.Array): The DDF before streaming, shape (9, NX, NY).
        loc (str): The location of the boundary where the bounce-back is applied.
                   Should be one of 'left', 'right', 'top', or 'bottom'.
        ux_wall (scalar or jax.Array of shape NX): The x-component of the wall velocity.
        uy_wall (scalar or jax.Array of shape NY): The y-component of the wall velocity.

    Returns:
        f (jax.Array): The DDF after applying the bounce-back boundary condition, shape (9, NX, NY).
    """

    if loc not in ["left", "right", "top", "bottom"]:
        raise ValueError(
            "Boundary location `loc` should be 'left', 'right', 'top', or 'bottom'."
        )

    if loc == "left":
        f = f.at[1, 0].set(f_before_stream[3, 0] + 2 / 3 * ux_wall)
        f = f.at[8, 0].set(f_before_stream[7, 0] - 1 / 6 * (-ux_wall - uy_wall))
        f = f.at[5, 0].set(f_before_stream[6, 0] - 1 / 6 * (-ux_wall + uy_wall))

    elif loc == "right":
        f = f.at[3, -1].set(f_before_stream[1, -1] - 2 / 3 * ux_wall)
        f = f.at[6, -1].set(f_before_stream[5, -1] - 1 / 6 * (ux_wall + uy_wall))
        f = f.at[7, -1].set(f_before_stream[8, -1] - 1 / 6 * (ux_wall - uy_wall))

    elif loc == "top":
        f = f.at[4, :, -1].set(f_before_stream[2, :, -1] - 2 / 3 * uy_wall)
        f = f.at[8, :, -1].set(f_before_stream[5, :, -1] - 1 / 6 * (ux_wall + uy_wall))
        f = f.at[7, :, -1].set(f_before_stream[6, :, -1] - 1 / 6 * (-ux_wall + uy_wall))

    elif loc == "bottom":
        f = f.at[2, :, 0].set(f_before_stream[4, :, 0] + 2 / 3 * uy_wall)
        f = f.at[6, :, 0].set(f_before_stream[7, :, 0] - 1 / 6 * (-ux_wall - uy_wall))
        f = f.at[5, :, 0].set(f_before_stream[8, :, 0] - 1 / 6 * (ux_wall - uy_wall))

    return f


def obstacle_bounce_back(f, mask):
    """Enforce a no-slip boundary using the Bounce Back scheme
    at any obstacles defined by a 2D mask, where True indicates
    the presence of an obstacle. 
    
    This can also be used to enforce no-slip boundarys at domain boundaries.

    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        mask (jax.Array of shape (NX, NY)): The mask indicating the obstacle.

    Returns:
        f (jax.Array of shape (9, NX, NY)): The DDF
            after enforcing the boundary condition.
    """

    return f.at[:, mask].set(f[:, mask][OPP_DIRS])
