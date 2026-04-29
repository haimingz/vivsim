r"""Shared helpers and boundary metadata for D2Q9 boundary conditions.

Direction numbering in this codebase follows:

    6   2   5
      \ | /
    3 - 0 - 1
      / | \
    7   4   8

The :class:`BoundarySpec` entries in ``BOUNDARY_SPEC`` encode the geometry and
population grouping for each domain boundary.

Example: left wall

    wall | fluid
         |
         |  wall          = (0,)
         |  neighbor      = (1,)
         |  in_dirs       = (1, 5, 8)
         |  out_dirs      = (3, 7, 6)
         |  tan_dirs      = (2, 4)
         |  pos_side_dirs = (2, 5, 6)
         |  neg_side_dirs = (4, 7, 8)
         |  normal_axis   = 0
         |  normal_sign   = 1

Meaning:
    - ``wall`` selects the boundary nodes in array indexing form
    - ``neighbor`` selects the first fluid line adjacent to the wall
    - ``in_dirs`` and ``out_dirs`` are defined with respect to the wall-normal direction
    - ``tan_dirs`` are the two pure axial directions tangent to the wall
    - ``pos_side_dirs`` and ``neg_side_dirs`` collect populations carrying positive or negative tangential momentum
    - ``normal_axis`` is ``0`` for left/right walls and ``1`` for bottom/top
    - ``normal_sign`` points from the wall into the fluid
"""

from typing import NamedTuple

import jax.numpy as jnp

from ..basic import get_velocity_correction


class BoundarySpec(NamedTuple):
    """Geometry and direction metadata for a single domain boundary."""

    wall: tuple
    neighbor: tuple
    in_dirs: tuple[int, int, int]
    out_dirs: tuple[int, int, int]
    tan_dirs: tuple[int, int]
    pos_side_dirs: tuple[int, int, int]
    neg_side_dirs: tuple[int, int, int]
    normal_sign: int
    normal_axis: int

    
BOUNDARY_SPEC = {
    "left": BoundarySpec(
        wall=(0,),
        neighbor=(1,),
        in_dirs=(1, 5, 8),
        out_dirs=(3, 7, 6),
        tan_dirs=(2, 4),
        pos_side_dirs=(2, 5, 6),
        neg_side_dirs=(4, 7, 8),
        normal_sign=1,
        normal_axis=0,
    ),
    "right": BoundarySpec(
        wall=(-1,),
        neighbor=(-2,),
        in_dirs=(3, 7, 6),
        out_dirs=(1, 5, 8),
        tan_dirs=(2, 4),
        pos_side_dirs=(2, 5, 6),
        neg_side_dirs=(4, 7, 8),
        normal_sign=-1,
        normal_axis=0,
    ),
    "top": BoundarySpec(
        wall=(slice(None), -1),
        neighbor=(slice(None), -2),
        in_dirs=(4, 7, 8),
        out_dirs=(2, 5, 6),
        tan_dirs=(1, 3),
        pos_side_dirs=(1, 5, 8),
        neg_side_dirs=(3, 7, 6),
        normal_sign=-1,
        normal_axis=1,
    ),
    "bottom": BoundarySpec(
        wall=(slice(None), 0),
        neighbor=(slice(None), 1),
        in_dirs=(2, 5, 6),
        out_dirs=(4, 7, 8),
        tan_dirs=(1, 3),
        pos_side_dirs=(1, 5, 8),
        neg_side_dirs=(3, 7, 6),
        normal_sign=1,
        normal_axis=1,
    ),
}


def get_boundary_shape(f, loc: str):
    """Return the number of nodes on the selected boundary."""
    spec = BOUNDARY_SPEC[loc]
    return f.shape[2] if spec.normal_axis == 0 else f.shape[1]


def broadcast_wall_values(f, loc: str, rho_wall=1, ux_wall=0, uy_wall=0):
    """Broadcast scalar wall data to the selected boundary length.

    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function.
        loc (str): Boundary location, one of ``'left'``, ``'right'``, ``'top'``,
            or ``'bottom'``.
        rho_wall: Scalar or array-like density prescribed on the wall.
        ux_wall: Scalar or array-like x-velocity prescribed on the wall.
        uy_wall: Scalar or array-like y-velocity prescribed on the wall.

    Returns:
        tuple: ``(rho_wall, u_wall)`` where ``rho_wall`` has shape ``(N,)`` and
        ``u_wall`` has shape ``(2, N)`` along the selected boundary.
    """
    shape = get_boundary_shape(f, loc)

    if jnp.isscalar(rho_wall):
        rho_wall = jnp.full(shape, rho_wall)
    if jnp.isscalar(ux_wall):
        ux_wall = jnp.full(shape, ux_wall)
    if jnp.isscalar(uy_wall):
        uy_wall = jnp.full(shape, uy_wall)

    return rho_wall, jnp.array([ux_wall, uy_wall])


def get_rho_wall_from_velocity(f, loc: str, ux_wall=0, uy_wall=0):
    """Compute wall density for a velocity boundary after streaming.

    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        loc (str): Boundary location, one of ``'left'``, ``'right'``, ``'top'``,
            or ``'bottom'``.
        ux_wall (scalar or jax.Array of shape N): The x-component of velocity.
        uy_wall (scalar or jax.Array of shape N): The y-component of velocity.

    Returns:
        scalar or jax.Array: The density at the wall.
    """
    normal_velocity, _ = get_signed_wall_velocity(loc, ux_wall=ux_wall, uy_wall=uy_wall)

    return _get_rho_wall_numerator(f, loc) / (1 - normal_velocity)


def get_wall_velocity_from_pressure(f, loc: str, rho_wall=1):
    """Compute wall velocity for a pressure boundary after streaming.

    The normal velocity is recovered from the prescribed wall density and the
    known distributions at the boundary. The tangential velocity is copied from
    the adjacent fluid node.

    Args:
        f (jax.Array of shape (9, NX, NY)): Discrete distribution function (DDF).
        loc (str): Boundary location, one of ``'left'``, ``'right'``, ``'top'``,
            or ``'bottom'``.
        rho_wall (scalar or jax.Array of shape N): The density at the wall.

    Returns:
        tuple: ``(ux_wall, uy_wall)`` recovered at the wall.
    """
    spec = BOUNDARY_SPEC[loc]
    neighbor = spec.neighbor
    ps0, ps1, ps2 = spec.pos_side_dirs
    ns0, ns1, ns2 = spec.neg_side_dirs

    rho_neighbor = jnp.sum(f[:, *neighbor], axis=0)
    normal_velocity_signed = 1 - _get_rho_wall_numerator(f, loc) / rho_wall
    normal_velocity = spec.normal_sign * normal_velocity_signed

    tangential_velocity = (
        f[ps0, *neighbor] - f[ns0, *neighbor]
        + f[ps1, *neighbor] - f[ns1, *neighbor]
        + f[ps2, *neighbor] - f[ns2, *neighbor]
    ) / rho_neighbor

    velocity = [None, None]
    velocity[spec.normal_axis] = normal_velocity
    velocity[1 - spec.normal_axis] = tangential_velocity
    return tuple(velocity)


def _get_rho_wall_numerator(f, loc: str):
    """Return the shared numerator used in wall density/velocity recovery."""
    spec = BOUNDARY_SPEC[loc]
    wall = spec.wall
    t0, t1 = spec.tan_dirs
    o0, o1, o2 = spec.out_dirs

    return (
        f[0, *wall] + f[t0, *wall] + f[t1, *wall]
        + 2 * (f[o0, *wall] + f[o1, *wall] + f[o2, *wall])
    )


def get_signed_wall_velocity(loc: str, ux_wall=0, uy_wall=0):
    """Return signed wall-normal and wall-tangential velocity components."""
    spec = BOUNDARY_SPEC[loc]
    normal_velocity = ux_wall if spec.normal_axis == 0 else uy_wall
    tangential_velocity = uy_wall if spec.normal_axis == 0 else ux_wall
    return spec.normal_sign * normal_velocity, spec.normal_sign * tangential_velocity


def get_corrected_wall_velocity(ux_wall, uy_wall, rho_wall=1, gx_wall=0, gy_wall=0):
    """Apply body force correction to the wall velocity.

    When body forces (e.g. gravity) are present in the simulation, the velocity
    imposed at the boundary must be shifted by half a force step to maintain
    second-order accuracy consistent with the rest of the domain.  Call this
    helper before passing the wall velocity to any boundary condition function.

    Args:
        ux_wall (scalar or jax.Array): The x-component of the prescribed wall velocity.
        uy_wall (scalar or jax.Array): The y-component of the prescribed wall velocity.
        rho_wall (scalar or jax.Array): The density at the wall (default 1).
        gx_wall (scalar or jax.Array): The x-component of the body force at the wall (default 0).
        gy_wall (scalar or jax.Array): The y-component of the body force at the wall (default 0).

    Returns:
        tuple: ``(ux_corrected, uy_corrected)`` - the velocity components after
        subtracting the half-step force contribution.
    """
    ux_corrected = ux_wall - get_velocity_correction(gx_wall, rho_wall)
    uy_corrected = uy_wall - get_velocity_correction(gy_wall, rho_wall)
    return ux_corrected, uy_corrected


def wrap_velocity(boundary_fn):
    """Build a velocity boundary function from a boundary kernel.

    The input ``boundary_fn`` is the core kernel with signature
    ``boundary_fn(f, loc, rho_wall, ux_wall, uy_wall)``. The returned function
    exposes the usual velocity-boundary interface
    ``wrapped(f, loc, ux_wall=0, uy_wall=0)``.

    It computes the missing wall density from the prescribed wall velocity using
    ``get_rho_wall_from_velocity(...)``, then passes the completed
    ``(rho_wall, ux_wall, uy_wall)`` tuple into the kernel.
    """

    def wrapped(f, loc: str, ux_wall=0, uy_wall=0):
        rho_wall = get_rho_wall_from_velocity(f, loc, ux_wall=ux_wall, uy_wall=uy_wall)
        return boundary_fn(f, loc, rho_wall=rho_wall, ux_wall=ux_wall, uy_wall=uy_wall)

    return wrapped


def wrap_pressure(boundary_fn):
    """Build a pressure boundary function from a boundary kernel.

    The input ``boundary_fn`` is the core kernel with signature
    ``boundary_fn(f, loc, rho_wall, ux_wall, uy_wall)``. The returned function
    exposes the usual pressure-boundary interface
    ``wrapped(f, loc, rho_wall=1)``.

    It computes the missing wall velocity from the prescribed wall density using
    ``get_wall_velocity_from_pressure(...)``, then passes the completed
    ``(rho_wall, ux_wall, uy_wall)`` tuple into the kernel.
    """

    def wrapped(f, loc: str, rho_wall=1):
        ux_wall, uy_wall = get_wall_velocity_from_pressure(f, loc, rho_wall=rho_wall)
        return boundary_fn(f, loc, rho_wall=rho_wall, ux_wall=ux_wall, uy_wall=uy_wall)

    return wrapped


def wrap_force_corrected(boundary_fn):
    """Build a force-corrected boundary function from a boundary kernel.

    The input ``boundary_fn`` is the core kernel with signature
    ``boundary_fn(f, loc, rho_wall, ux_wall, uy_wall)``. The returned function
    exposes the interface
    ``wrapped(f, loc, rho_wall=1, ux_wall=0, uy_wall=0, gx_wall=0, gy_wall=0)``.

    It applies the half-force wall-velocity correction through
    ``get_corrected_wall_velocity(...)`` before passing the corrected
    ``(rho_wall, ux_wall, uy_wall)`` into the kernel.
    """

    def wrapped(f, loc: str, rho_wall=1, ux_wall=0, uy_wall=0, gx_wall=0, gy_wall=0):
        ux_wall, uy_wall = get_corrected_wall_velocity(
            ux_wall, uy_wall, rho_wall=rho_wall, gx_wall=gx_wall, gy_wall=gy_wall
        )
        return boundary_fn(f, loc, rho_wall=rho_wall, ux_wall=ux_wall, uy_wall=uy_wall)

    return wrapped
