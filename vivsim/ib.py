"""This file contains the functions associated with the immersed boundary (IB) method.

The IB method is used to transfer the force between the fluid (discretized into a regular Eulerian lattice) 
and the object (represented by a set of Lagrangian markers). 
The markers can move freely in the lattice, meaning the markers and lattice are not aligned. 

In this file, the following notations are used:
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
from .lbm import get_u_correction

# ----------------- Kernel functions -----------------


def kernel_func2(distance):
    """kernel function of range 2
    
    Args:
        distance (scalar or ndarray): The distance between the marker and the lattice.
    
    Returns:
        out (scalar or ndarray): The kernel function value. 
    """
    
    return jnp.where(jnp.abs(distance) <= 1, 1 - jnp.abs(distance), 0)


def kernel_func3(distance):
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


def kernel_func4(distance):
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


def get_kernels(x_markers, y_markers, x_lattice, y_lattice, kernel_function):
    """Generate the kernels for all the markers.
    
    Args:
        x_markers, y_markers (ndarray of shape (N_MARKER)): The coordinates of markers.
        x_lattice, y_lattice (ndarray of shape (NX, NY)): The coordinates of the lattice.
        kernel (callable): The kernel function. Available options: 
            kernel_func2, kernel_func3, kernel_func4.
    
    Returns:
        out (ndarray of shape (N_MARKER, NX, NY))： The kernel function values.
    """
    return (kernel_function(x_lattice[None, ...] - x_markers[:, None, None]) \
          * kernel_function(y_lattice[None, ...] - y_markers[:, None, None]))


# ----------------- Fluid Velocity Interpolation -----------------


def interpolate_u_marker(u, kernel):
    """Interpolate the fluid velocity (u) at a marker.
    
    Args:
        u: The velocity field with shape (2, NX, NY).
        kernel: The kernel function with shape (NX, NY).
    
    Returns:
        The interpolated fluid velocity at a marker with shape (2).
    """
    
    return jnp.einsum("xy,dxy->d", kernel, u)


def interpolate_u_markers(u, kernels):
    """Interpolate the fluid velocity at all markers.
    
    Args:
        u: The velocity field with shape (2, NX, NY).
        kernels: The kernel functions with shape (N_MARKER, NX, NY).
    
    Returns:
        The interpolated fluid velocity at all markers with shape (N_MARKER, 2).
    """
    
    return jnp.einsum("nxy,dxy->nd", kernels, u)


# ----------------- Compute IB Force to the fluid -----------------


def get_delta_g_marker(v_marker, u_marker):
    """Compute the correction force to the fluid.
    According to g = 2 * rho * (v - u) / dt
    
    Args:
        v_marker: The velocity of the object at the marker with shape (2).
        u_marker: The interpolated velocity of the fluid at the marker with shape (2).
    
    Returns:
        The correction force with shape (2).
    """
    
    return 2 * (v_marker - u_marker)


def get_delta_g_markers(v_markers, u_markers):
    """Compute the correction force to the fluid.    
    According to g = 2 * rho * (v - u) / dt
    
    Args:
        v_markers: The velocity of the object at the markers with shape (N_MARKER, 2) or (2).
        u_markers: The interpolated velocity of the fluid at the markers with shape (N_MARKER, 2).
    
    Returns:
        The correction forces with shape (N_MARKER, 2).
    """
    if v_markers.ndim == 1:
        return 2 * (v_markers[None,...] - u_markers)
       
    return 2 * (v_markers - u_markers)


def distribute_g_marker(g_marker, kernel):
    """Spread the force to the lattice.
    
    Args:
        force: The force vector with shape (2).
        kernel: The kernel function with shape (NX, NY).
    
    Returns:
        The force field with shape (2, NX, NY)."""
    
    return kernel * g_marker[:, None, None]


def distribute_g_markers(g_markers, kernels):
    """Spread the force to the lattice.
    
    Args:
        forces: The force vector with shape (N_MARKER, 2).
        kernels: The kernel functions with shape (N_MARKER, NX, NY).
    
    Returns:
        The force field with shape (2, NX, NY)."""
    
    return jnp.einsum("nd,nxy->dxy", g_markers, kernels)


# ----------------- Object -> Markers -----------------


def update_markers_coords_2dof(X_MARKERS, Y_MARKERS, d):
    """update the real-time position of the markers (consider 2DOF)
    
    Args:
        X_MARKERS, Y_MARKERS: The original coordinates of the markers with shape (N_MARKER).
        d: The displacement of the object with shape (2).
    
    Returns:
        x_markers, y_markers: The updated coordinates with shape (N_MARKER).
    """
    
    x_markers = X_MARKERS + d[0] 
    y_markers = Y_MARKERS + d[1] 
    return x_markers, y_markers


def update_markers_coords_3dof(X_MARKERS, Y_MARKERS, X_CENTER, Y_CENTER, d):
    """update the real-time position of the markers (consider 3DOF)
    
    Args:
        X_MARKERS, Y_MARKERS: The original coordinates of the markers with shape (N_MARKER).
        X_CENTER, Y_CENTER: The coordinates of the object center.
        d: The displacement of the object with shape (3).
    
    Returns:
        x_markers, y_markers: The updated coordinates with shape (N_MARKER).
    """

    # # rotation_matrix (2x2)
    # rotation_matrix = jnp.array([[jnp.cos(d[2]), -jnp.sin(d[2])],
    #                              [jnp.sin(d[2]), jnp.cos(d[2])]])
    
    # # relative coords of the markers to the object center
    # coords_relative = jnp.vstack((X_MARKERS - X_CENTER, Y_MARKERS - Y_CENTER))
    
    # # rotate the relative coords
    # coords_relative_rotation = jnp.dot(rotation_matrix, coords_relative)
    
    # # global coords 
    # x_markers = coords_relative_rotation[0] + X_CENTER + d[0]
    # y_markers = coords_relative_rotation[1] + Y_CENTER + d[1]

    x_rel = X_MARKERS - X_CENTER
    y_rel = Y_MARKERS - Y_CENTER
    x_markers = x_rel * jnp.cos(d[2]) - y_rel * jnp.sin(d[2]) + d[0] + X_CENTER
    y_markers = x_rel * jnp.sin(d[2]) + y_rel * jnp.cos(d[2]) + d[1] + Y_CENTER
    
    return x_markers, y_markers


def update_markers_velocity_3dof(x_marker, y_marker, X_CENTER, Y_CENTER, d, v):
    """Compute the velocity of the markers.
    
    Args:
        v: The velocity of the object with shape (3).
        d: The displacement of the object with shape (d).
    
    Returns:
        The velocity of the markers with shape (N_MARKER, 2).
    """
    x_rel = x_marker - X_CENTER - d[0]
    y_rel = y_marker - Y_CENTER - d[1]
    
    v_markers = jnp.zeros((x_marker.shape[0], 2))
    v_markers = v_markers.at[:, 0].set(- v[2] * x_rel + v[0])
    v_markers = v_markers.at[:, 1].set(v[2] * y_rel + v[1])
    return v_markers


# ----------------- Solving scheme -----------------


def multi_direct_forcing(
    u, X, Y,  # fluid velocity field and mesh grid
    v_markers, x_markers, y_markers,  # markers velocity and coordinates
    X1=0, X2=-1, Y1=0, Y2=-1,  # the range of the IBM region
    N_ITER=3,
    ):
    """Implement the multi-direct forcing method.
    
    Args:
        u: The velocity field with shape (2, NX, NY).
        d: The displacement of the object with shape (2).
        v_markers: The velocity of the object with shape (N_MARKER, 2).
        X, Y: The mesh grid with shape (NX, NY).
        X_MARKERS, Y_MARKERS: The coordinates of the markers with shape (N_MARKER).
        X1, X2, Y1, Y2 (int): The range of the IBM region.
        N_ITER (int): The number of iterations for multi-direct forcing.
    
    Returns:
        h_markers: The hydrodynamic force to the markers with shape (N_MARKER, 2).
        g: The distributed IB force to the fluid with shape (2, NX, NY).
        u: The updated velocity field with shape (2, NX, NY).
    """
    
    # initialize the force terms
    h_markers = jnp.zeros((x_markers.shape[0], 2))  # hydrodynamic force to the markers
    g = jnp.zeros((2, X2 - X1, Y2 - Y1))  # distributed IB force to the fluid
    
   
    # calculate the kernel functions for all markers
    kernels = get_kernels(x_markers, y_markers, X[X1:X2, Y1:Y2], Y[X1:X2, Y1:Y2], kernel_func4)
    
    for _ in range(N_ITER):
        
        # velocity interpolation
        u_markers = interpolate_u_markers(u[:, X1:X2, Y1:Y2], kernels)
        
        # compute correction force
        delta_g_markers = get_delta_g_markers(v_markers, u_markers)
        delta_g = distribute_g_markers(delta_g_markers, kernels)
        
        # velocity correction
        u = u.at[:, X1:X2, Y1:Y2].add(get_u_correction(delta_g))
        
        # accumulate the corresponding correction force to the markers and the fluid
        h_markers += - delta_g_markers
        g += delta_g
        
    return  h_markers, g, u


# ----------------- Markers -> Object -----------------


def calculate_force_obj(h_markers, marker_distance):
    """Compute the total force to the object.
    
    Args:
        h_markers: The hydrodynamic force to the markers with shape (N_MARKER, 2).
        marker_distance: The distance between the markers.
    
    Returns:
        The total force to the object with shape (2).
    """
    
    return jnp.sum(h_markers, axis=0) * marker_distance


def calculate_torque_obj(x_markers, y_markers, X_CENTER, Y_CENTER, d,  h_markers, marker_distance):
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
    
    coord_rel = jnp.vstack((x_rel, y_rel))
    torque = jnp.sum(jnp.cross(coord_rel, (h_markers * marker_distance).T, axis=0))
    return torque

