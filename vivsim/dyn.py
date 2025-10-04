import jax.numpy as jnp



def newmark(a, v, d, h, m, k, c, dt=1, gamma=0.5, beta=0.25):
    """Newmark-beta method for dynamics
    
    This unified method handles both scalar (1-DOF) and matrix (multi-DOF) systems
    automatically based on the input types.
    
    Args:
        a (float or array): acceleration at time t
        v (float or array): velocity at time t
        d (float or array): displacement at time t
        h (float or array): external force at time t
        m (float or array): mass (matrix for multi-DOF)
        k (float or array): stiffness (matrix for multi-DOF)
        c (float or array): damping (matrix for multi-DOF)
        dt (float): time step
        gamma (float): gamma parameter for Newmark-beta method
        beta (float): beta parameter for Newmark-beta method
    
    Returns:
        a_next (float or array): acceleration at time t+1
        v_next (float or array): velocity at time t+1
        d_next (float or array): displacement at time t+1
    """
    
    c1, c2 = gamma * dt, beta * dt ** 2
    
    # Predictor step
    v1 = v + dt * (1 - gamma) * a
    v2 = d + dt * v + dt ** 2 * (0.5 - beta) * a
    
    # Check if inputs are matrices (multi-DOF) or scalars (single-DOF)
    if jnp.ndim(m) > 0:  
        # Multi-DOF case (matrix inputs)
        a_next = jnp.linalg.solve(
            m + c1 * c + c2 * k,
            h - jnp.dot(c, v1) - jnp.dot(k, v2)
        )
    else:  
        # Single-DOF case (scalar inputs)
        # Direct calculation for scalar case
        a_next = (h - c * v1 - k * v2) / (m + c1 * c + c2 * k)
    
    # Corrector step
    v_next = c1 * a_next + v1
    d_next = c2 * a_next + v2
    
    return a_next, v_next, d_next


# Backward compatibility functions
def newmark_2dof(a, v, d, h, m, k, c, dt=1, gamma=0.5, beta=0.25):
    """Legacy function for 2-DOF systems. Use newmark_beta instead."""
    return newmark(a, v, d, h, m, k, c, dt, gamma, beta)


def newmark_3dof(a, v, d, h, m, k, c, dt=1, gamma=0.5, beta=0.25):
    """Legacy function for 3-DOF systems. Use newmark_beta instead."""
    return newmark(a, v, d, h, m, k, c, dt, gamma, beta)


# ----------------- Kinetics from Rigid Object to Markers -----------------


def get_markers_coords_2dof(x_markers_init, y_markers_init, d):
    """update the real-time position of the markers (without rotation)
    
    Args:
        x_markers_init, y_markers_init (ndarray of shape (N_MARKER)): The initial coordinates of the markers.
        d (ndarray of shape (2)): The instantaneous displacement of the object.
    
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
        d (ndarray of shape (3)): The instantaneous displacement of the object.
    
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