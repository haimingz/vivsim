"""This file implements the immersed boundary method (IBM) in the lattice Boltzmann method framework.

The IB method is used to transfer the force between the fluid (discretized into a regular Eulerian lattice) 
and the object (represented by a set of Lagrangian markers). 
The markers can move freely in the lattice, meaning the markers and lattice are not aligned. 

Key variables:
- u: The velocity field of the fluid (distributed).
- u_marker(s): The interpolated velocity of the fluid at the marker(s).
- v: The velocity of the object (as a whole body).
- v_marker(s): The velocity of the object at marker(s).
- g: The force that the object applied to the fluid (distributed)
- g_marker(s): The force that the object applied to the fluid at marker(s).
- h: The force that the fluid applied to the object (as a whole body)
- h_marker(s): The force that the fluid applied to the object at marker(s)

"""

import jax.numpy as jnp


# ----------------- Kinetics from Object to Markers -----------------


def update_markers_coords_2dof(X_MARKERS, Y_MARKERS, d):
    """update the real-time position of the markers (ignoring rotation)
    
    Args:
        X_MARKERS, Y_MARKERS (ndarray of shape (N_MARKER)): The original coordinates of the markers.
        d (ndarray of shape (1) or (3)): The displacement of the object.
    
    Returns:
        x_markers, y_markers (ndarray of shape (N_MARKER)): The updated coordinates.
    """
    
    x_markers = X_MARKERS + d[0]
    y_markers = Y_MARKERS + d[1]
    return x_markers, y_markers


def update_markers_coords_3dof(X_MARKERS, Y_MARKERS, X_CENTER, Y_CENTER, d):
    """update the real-time position of the markers (consider rotation)
    
    Args:
        X_MARKERS, Y_MARKERS (ndarray of shape (N_MARKER)): The original coordinates of the markers.
        X_CENTER, Y_CENTER (scaler): The original coordinates of the object center.
        d (ndarray of shape (2)): The displacement of the object.
    
    Returns:
        x_markers, y_markers (ndarray of shape (N_MARKER)): The updated coordinates.
    """
    x_rel = X_MARKERS - X_CENTER
    y_rel = Y_MARKERS - Y_CENTER
    x_markers = X_CENTER + d[0] + x_rel * jnp.cos(d[2]) - y_rel * jnp.sin(d[2]) 
    y_markers = Y_CENTER + d[1] + x_rel * jnp.sin(d[2]) + y_rel * jnp.cos(d[2]) 
    
    return x_markers, y_markers


def update_markers_velocity_3dof(x_markers, y_markers, X_CENTER, Y_CENTER, d, v):
    """Compute the velocity of the markers considering rotation.
    
    Args:
        x_markers, y_marker (ndarray of shape (N_MARKER)): instantaneous coordinates of the markers.
        X_CENTER, Y_CENTER (scaler): The original coordinates of the object center.
        v: The velocity of the object with shape (2).
        d: The displacement of the object with shape (dim).
    
    Returns:
        The velocity of the markers with shape (N_MARKER, 1).
    """
    x_rel = x_markers - X_CENTER - d[0]
    y_rel = y_markers - Y_CENTER - d[1]
    
    v_markers = jnp.zeros((x_markers.shape[0], 2))
    v_markers = v_markers.at[:, 0].set(v[0] - v[2] * y_rel)
    v_markers = v_markers.at[:, 1].set(v[1] + v[2] * x_rel)
    return v_markers



# ----------------- Kernel functions -----------------


def kernel_range2(distance):
    """kernel function of range 2
    
    Args:
        distance (scalar or ndarray): The distance between the marker and the lattice.
    
    Returns:
        out (scalar or ndarray): The kernel function value. 
    """
    
    return jnp.where(jnp.abs(distance) <= 1, 1 - jnp.abs(distance), 0)


def kernel_range3(distance):
    """kernel function of range 3
    
    Args:
        distance (scalar or ndarray): The distance between the marker and the lattice.
    
    Returns:
        out (scalar or ndarray): The kernel function value. 
    """
    
    distance = jnp.abs(distance)
    return jnp.where(
        distance > 1.5,
        0,
        jnp.where(
            distance < 0.5,
            (1 + jnp.sqrt(1 - 3 * distance**2)) / 3,
            (5 - 3 * distance - jnp.sqrt(-2 + 6 * distance - 3 * distance**2)) / 6,
        ),
    )


def kernel_range4(distance):
    """kernel function of range 4
    
    Args:
        distance (scalar or ndarray): The distance between the marker and the lattice.
    
    Returns:
        out (scalar or ndarray): The kernel function value. 
    """
    
    distance = jnp.abs(distance)
    return jnp.where(
        distance > 2,
        0,
        jnp.where(
            distance < 1,
            (3 - 2 * distance + jnp.sqrt(1 + 4 * distance - 4 * distance ** 2)) / 8,
            (5 - 2 * distance - jnp.sqrt(- 7 + 12 * distance - 4 * distance ** 2)) / 8,
        ),
    )


def get_kernels(x_markers, y_markers, x_lattice, y_lattice, kernel_func):
    """Generate the kernels for all the markers.
    
    Args:
        x_markers, y_markers (ndarray of shape (N_MARKER)): The coordinates of markers.
        x_lattice, y_lattice (ndarray of shape (NX, NY)): The coordinates of the lattice.
        kernel_func (callable): The kernel function. Available options: 
            kernel_range2, kernel_range3, kernel_range4.
    
    Returns:
        out (ndarray of shape (N_MARKER, NX, NY))： The kernel function values.
    """
    return (kernel_func(x_lattice[None, ...] - x_markers[:, None, None]) \
          * kernel_func(y_lattice[None, ...] - y_markers[:, None, None]))


# ----------------- Core IB calculation -----------------


def interpolate_u_markers(u, kernels):
    """Interpolate the fluid velocity at all markers.
    
    Args:
        u (ndarray of shape (2, NX, NY)): The velocity field of fluid.
        kernels (ndarray of shape (N_MARKER, NX, NY)): The kernel functions.
    
    Returns:
        out (ndarray of shape (N_MARKER, 2)): The interpolated fluid velocity at markers.
    """
    
    return jnp.einsum("nxy,dxy->nd", kernels, u)


def get_g_markers_needed(v_markers, u_markers, marker_distance, rho=1):
    """Compute the needed forces required to enforce no-slip boundary at markers.
    according to g = 2 * rho * (v - u) / dt.
    
    Args:
        v_markers (ndarray of shape (N_MARKER, 2) or (2)): The velocity of the markers.
        u_markers (ndarray of shape (N_MARKER, 2)): The interpolated fluid velocity at the markers.
        rho (scaler or ndarray of shape (N_MARKER)): The density of the fluid.
    
    Returns:
        out (ndarray of shape (N_MARKER, 2)): The needed correction forces.
    """
    if v_markers.ndim == 1:
        return 2 * (v_markers[None,...] - u_markers)
       
    return 2 * (v_markers - u_markers) * rho * marker_distance


def spread_g_needed(g_markers, kernels):
    """Spread the correction force that act only at the markers to the fluid.
    
    Args:
        g_markers (ndarray of shape (N_MARKER, 2)): The correction force vector.
        kernels (ndarray of shape (N_MARKER, NX, NY)): The kernel functions.
    
    Returns:
        out (ndarray of shape (2, NX, NY)): The force field applied to the fluid.
    """
    
    return jnp.einsum("nd,nxy->dxy", g_markers, kernels)


# ----------------- Markers -> Object -----------------


def calculate_force_obj(h_markers):
    """Compute the total force to the object.
    
    Args:
        h_markers: The hydrodynamic force to the markers with shape (N_MARKER, 2).
        marker_distance: The distance between the markers.
    
    Returns:
        The total force to the object with shape (2).
    """
    
    return jnp.sum(h_markers, axis=0)


def calculate_torque_obj(x_markers, y_markers, X_CENTER, Y_CENTER, d, h_markers):
    """Calculate the torque applied to the object.
    
    Args:
        x_markers, y_markers: The coordinates of the markers with shape (N_MARKER).
        X_CENTER, Y_CENTER: The coordinates of the object center.
        h_markers: The distributed force to the fluid with shape (N_MARKER, 2).
        marker_distance: The distance between the markers.
    
    Returns:
        The torque (scalar).
    """
    
    x_rel = x_markers - X_CENTER - d[0]
    y_rel = y_markers - Y_CENTER - d[1]
    
    return jnp.sum(x_rel * h_markers[:, 1] - y_rel * h_markers[:, 0])

