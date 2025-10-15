"""
This module implements the characteristic boundary condition (CBC) for non-reflective
boundaries in a 2D D2Q9 lattice.

The CBC calculates the boundary density and velocity for the next time step based on the
current density and velocity at the boundary, as well as their spatial derivatives. The
"""



import jax.numpy as jnp


def boundary_characteristic(rho, u, dir='right'):
    """Non-reflective Characteristic Boundary Condition for D2Q9 LBM (lattice units).
    
    dx = dt = 1, cs = 1/sqrt(3).
    
    Args:
        rho: jnp.ndarray of shape (NX, NY)
        u: jnp.ndarray of shape (2, NX, NY)
        dir: one of {'left', 'right', 'top', 'bottom'}
    
    Returns:
        rho_next: boundary density (1D array)
        u_next: boundary velocity (2, N)
    """
    cs = 1 / jnp.sqrt(3)
    cs2 = 1 / 3

    if dir == 'right':
        rho_b1, rho_b2, rho_b3 = rho[-1, :], rho[-2, :], rho[-3, :]
        ux_b1, ux_b2, ux_b3 = u[0, -1, :], u[0, -2, :], u[0, -3, :]
        uy_b1 = u[1, -1, :]

        drho_dx = (3*rho_b1 - 4*rho_b2 + rho_b3) / 2
        dux_dx  = (3*ux_b1 - 4*ux_b2 + ux_b3) / 2

        L_out = (ux_b1 + cs) * (dux_dx + cs/rho_b1 * drho_dx)
        L_in  = (ux_b1 - cs) * (dux_dx - cs/rho_b1 * drho_dx)

        L_in_new = 0.0  # Non-reflective boundary (outflow)

        rho_next = rho_b1 - 0.5 * rho_b1 / cs * (L_out + L_in_new)
        ux_next  = ux_b1 - 0.5 * (L_out - L_in_new)
        uy_next  = uy_b1  # zero-gradient assumption

    elif dir == 'left':
        rho_b1, rho_b2, rho_b3 = rho[0, :], rho[1, :], rho[2, :]
        ux_b1, ux_b2, ux_b3 = u[0, 0, :], u[0, 1, :], u[0, 2, :]
        uy_b1 = u[1, 0, :]

        drho_dx = (-3*rho_b1 + 4*rho_b2 - rho_b3) / 2
        dux_dx  = (-3*ux_b1 + 4*ux_b2 - ux_b3) / 2

        L_out = (ux_b1 - cs) * (dux_dx - cs/rho_b1 * drho_dx)
        L_in  = (ux_b1 + cs) * (dux_dx + cs/rho_b1 * drho_dx)

        L_in_new = 0.0

        rho_next = rho_b1 - 0.5 * rho_b1 / cs * (L_out + L_in_new)
        ux_next  = ux_b1 - 0.5 * (L_out - L_in_new)
        uy_next  = uy_b1

    elif dir == 'top':
        rho_b1, rho_b2, rho_b3 = rho[:, -1], rho[:, -2], rho[:, -3]
        uy_b1, uy_b2, uy_b3 = u[1, :, -1], u[1, :, -2], u[1, :, -3]
        ux_b1 = u[0, :, -1]

        drho_dy = (3*rho_b1 - 4*rho_b2 + rho_b3) / 2
        duy_dy  = (3*uy_b1 - 4*uy_b2 + uy_b3) / 2

        L_out = (uy_b1 + cs) * (duy_dy + cs/rho_b1 * drho_dy)
        L_in  = (uy_b1 - cs) * (duy_dy - cs/rho_b1 * drho_dy)

        L_in_new = 0.0

        rho_next = rho_b1 - 0.5 * rho_b1 / cs * (L_out + L_in_new)
        uy_next  = uy_b1 - 0.5 * (L_out - L_in_new)
        ux_next  = ux_b1

    elif dir == 'bottom':
        rho_b1, rho_b2, rho_b3 = rho[:, 0], rho[:, 1], rho[:, 2]
        uy_b1, uy_b2, uy_b3 = u[1, :, 0], u[1, :, 1], u[1, :, 2]
        ux_b1 = u[0, :, 0]

        drho_dy = (-3*rho_b1 + 4*rho_b2 - rho_b3) / 2
        duy_dy  = (-3*uy_b1 + 4*uy_b2 - uy_b3) / 2

        L_out = (uy_b1 - cs) * (duy_dy - cs/rho_b1 * drho_dy)
        L_in  = (uy_b1 + cs) * (duy_dy + cs/rho_b1 * drho_dy)

        L_in_new = 0.0

        rho_next = rho_b1 - 0.5 * rho_b1 / cs * (L_out + L_in_new)
        uy_next  = uy_b1 - 0.5 * (L_out - L_in_new)
        ux_next  = ux_b1

    else:
        raise ValueError("dir must be one of: 'left', 'right', 'top', 'bottom'")

    # Return consistent shape
    if dir in ['left', 'right']:
        u_next = jnp.stack([ux_next, uy_next])
    else:
        u_next = jnp.stack([ux_next, uy_next], axis=0)

    return rho_next, u_next
