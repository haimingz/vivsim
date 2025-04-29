"""This file implements the immersed boundary method (IBM) in the lattice Boltzmann method framework.

The IBM is used to transfer force between fluid and solid object. 
The fluid-solid interface is represented by a set of Lagrangian markers that can move freely. 

Key variables:
- u: The velocity field of the fluid.
- u_markers: The interpolated velocity of the fluid at the lagrangian markers.
- v: The velocity of the object.
- v_markers: The velocity of the lagrangian markers on the fluid-structure interface.
- g: The force that the object applied to the fluid.
- g_markers: The force that the object applied to the fluid at the lagrangian markers.
- h: The force that the fluid applied to the object.
- h_markers: The force that the fluid applied to the object at the lagrangian markers

"""

import jax.numpy as jnp


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


def get_kernels(x_markers, y_markers, x, y, kernel_func):
    """Generate a stack of kernels for all the markers. The kernels are to be
    used in future interpolation and spreading operations.
    
    Args:
        x_markers, y_markers (ndarray of shape (N_MARKER)): The coordinates of lagrangian markers.
        x, y (ndarray of shape (NX, NY)): The coordinates of the lattice.
        kernel_func (callable): The kernel function. Available options: 
            kernel_range2, kernel_range3, kernel_range4.
    
    Returns:
        out (ndarray of shape (N_MARKER, NX, NY))： The stacked kernel functions.
    """
    return (kernel_func(x[None, ...] - x_markers[:, None, None]) \
          * kernel_func(y[None, ...] - y_markers[:, None, None]))


# ----------------- Core IB calculation -----------------


def multi_direct_forcing(u, x, y, v_markers, x_markers, y_markers, 
                         n_markers, seg_len_markers, n_iter=5, kernel_func=kernel_range4):
    """Multi-direct forcing method to enforce no-slip boundary at markers.
    
    Args:
        u (ndarray of shape (2, NX, NY)): The fluid velocity field.
        x, y (ndarray of shape (NX, NY)): The coordinates of the Eulerian points.
        v_markers (ndarray of shape (2)): The solid velocity at the markers.
        x_markers, y_markers (ndarray of shape (n_markers)): The coordinates of the Lagrangian markers.
        n_markers (int): The number of markers.
        seg_len_markers (scalar or ndarray of shape (n_markers)): The segment length of markers.
        n_iter (int): The number of iterations for multi-direct forcing.
        kernel_func (callable): The kernel function defining how the interface is diffused. 
            Available options: ib.kernel_range2, ib.kernel_range3, ib.kernel_range4.
    
    Returns:
        g (ndarray of shape (9, NX, NY)): The force density field applied to the fluid.
        h_markers (ndarray of shape (n_markers, 2)): The forces applied to individual markers.    
    """
        
    g = jnp.zeros_like(u)
    h_markers = jnp.zeros((n_markers, 2))
    kernels = get_kernels(x_markers, y_markers, x, y, kernel_func)
    for _ in range(n_iter):        
        # velocity at markers
        u_markers = jnp.einsum("dxy,nxy->nd", u, kernels) # fluid velocity at markers
        u_markers_diff = v_markers - u_markers  # fluid and solid velocity difference at markers 
        
        # force to fluid
        g_markers_correction = u_markers_diff * seg_len_markers * 2 # required force correction to fluid at markers
        g_correction = jnp.einsum("nd,nxy->dxy", g_markers_correction, kernels) # required force to fluid at lattice points
        g += g_correction # accumulate required force to fluid at lattice points        
        
        # force to solid
        h_markers -= u_markers_diff * seg_len_markers  # accumulate required force to solid at markers
        
        # update velocity
        u += g_correction * 0.5 # correct fluid velocity at lattice points (Guo's forcing scheme)
    
    return g, h_markers


def implicit(u, x, y, v_markers, x_markers, y_markers, 
             seg_len_markers,  kernel_func=kernel_range4):

    kernels = get_kernels(x_markers, y_markers, x, y, kernel_func) 
    
    # velocity at markers
    u_markers = jnp.einsum("dxy,nxy->nd", u, kernels) # fluid velocity at markers
    u_markers_diff = v_markers - u_markers # fluid and solid velocity difference at markers
    
    # required velocity correction
    A = jnp.einsum("mxy,nxy->mn", kernels, kernels)
    u_markers_correction = jnp.linalg.solve(A, u_markers_diff) # required fluid velocity correction at markers
    u_correction = jnp.einsum("nd,nxy->dxy", u_markers_correction, kernels) # required fluid velocity correction at lattice points
    
    # required force correction
    h_markers = - u_markers_correction * seg_len_markers # required force to solid at markers
    g = u_correction * 2 * seg_len_markers # required force to fluid at lattice points
    
    return g, h_markers

# ----------------- Kinetics from Rigid Object to Markers -----------------


def get_markers_coords_2dof(x_markers_init, y_markers_init, d):
    """update the real-time position of the markers (without rotation)
    
    Args:
        x_markers_init, y_markers_init (ndarray of shape (N_MARKER)): The initial coordinates of the markers.
        d (ndarray of shape (1) or (3)): The instantaneous displacement of the object.
    
    Returns:
        x_markers, y_markers (ndarray of shape (N_MARKER)): The updated coordinates.
    """
    
    x_markers = x_markers_init + d[0]
    y_markers = y_markers_init + d[1]
    return x_markers, y_markers


def get_markers_coords_3dof(x_markers_init, y_markers_init, x_center_init, y_center_init, d):
    """update the real-time position of the markers (consider rotation)
    
    Args:
        x_markers_init, y_markers_init (ndarray of shape (N_MARKER)): The initial coordinates of the markers.
        x_center_init, y_center_init (scaler): The original coordinates of the object center.
        d (ndarray of shape (2)): The instantaneous displacement of the object.
    
    Returns:
        x_markers, y_markers (ndarray of shape (N_MARKER)): The updated coordinates.
    """
    x_rel = x_markers_init - x_center_init
    y_rel = y_markers_init - y_center_init
    x_markers = x_center_init + d[0] + x_rel * jnp.cos(d[2]) - y_rel * jnp.sin(d[2]) 
    y_markers = y_center_init + d[1] + x_rel * jnp.sin(d[2]) + y_rel * jnp.cos(d[2]) 
    
    return x_markers, y_markers


def get_markers_velocity_3dof(x_markers, y_markers, x_center_init, y_center_init, d, v):
    """Compute the velocity of the markers considering rotation.
    
    Args:
        x_markers, y_markers (ndarray of shape (N_MARKER)): instantaneous coordinates of the markers.
        x_center_init, y_center_init (scaler): The initial coordinates of the object center.
        d: The instantaneous displacement of the object with shape (dim).
        v: The instantaneous velocity of the object with shape (2).
    
    Returns:
        The velocity of the markers with shape (N_MARKER, 1).
    """
    x_rel = x_markers - x_center_init - d[0]
    y_rel = y_markers - y_center_init - d[1]
    
    v_markers = jnp.zeros((x_markers.shape[0], 2))
    v_markers = v_markers.at[:, 0].set(v[0] - v[2] * y_rel)
    v_markers = v_markers.at[:, 1].set(v[1] + v[2] * x_rel)
    return v_markers


# ----------------- Markers -> Rigid Object -----------------


def get_force_to_obj(h_markers):
    """Compute the total force to the object.
    
    Args:
        h_markers: The hydrodynamic force to the markers with shape (N_MARKER, 2).
    
    Returns:
        The total force to the object with shape (2).
    """
    
    return jnp.sum(h_markers, axis=0)


def get_torque_to_obj(x_markers, y_markers, x_center_init, y_center_init, d, h_markers):
    """Calculate the torque applied to the object.
    
    Args:
        x_markers, y_markers: The coordinates of the markers with shape (N_MARKER).
        x_center_init, y_center_init: The initial coordinates of the object center.
        h_markers: The distributed force to the fluid with shape (N_MARKER, 2).
    
    Returns:
        The torque (scalar).
    """
    
    x_rel = x_markers - (x_center_init + d[0])
    y_rel = y_markers - (y_center_init + d[1])
    
    return jnp.sum(x_rel * h_markers[:, 1] - y_rel * h_markers[:, 0])

