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
        x_center_init, y_center_init (scalar): The original coordinates of the object center.
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
        x_center_init, y_center_init (scalar): The initial coordinates of the object center.
        d: The instantaneous displacement of the object with shape (dim).
        v: The instantaneous velocity of the object with shape (2).
    
    Returns:
        The velocity of the markers with shape (N_MARKER, 2).
    """
    x_rel = x_markers - x_center_init - d[0]
    y_rel = y_markers - y_center_init - d[1]
    
    v_markers = jnp.stack([v[0] - v[2] * y_rel, v[1] + v[2] * x_rel], axis=-1)

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
# ----------------- Flexible Object Dynamics -----------------

def vfife_beam(x, v, theta, omega, f_ext, m_ext, m, J, EA, EI, l0, dt, damping=0.0):
    """Vector Form Intrinsic Finite Element (VFIFE) method for 2D flexible beams.

    This function computes one time step of the explicit dynamics for a 2D beam
    using the VFIFE method, which is well-suited for large deformations and JAX
    parallelization.

    Args:
        x (ndarray of shape (N, 2)): Nodal positions at time t.
        v (ndarray of shape (N, 2)): Nodal velocities at time t.
        theta (ndarray of shape (N,)): Nodal rotations at time t.
        omega (ndarray of shape (N,)): Nodal angular velocities at time t.
        f_ext (ndarray of shape (N, 2)): External forces on nodes (e.g., fluid forces).
        m_ext (ndarray of shape (N,)): External moments on nodes.
        m (ndarray of shape (N,)): Node masses.
        J (ndarray of shape (N,)): Node moments of inertia.
        EA (float or ndarray of shape (N-1,)): Axial stiffness of elements.
        EI (float or ndarray of shape (N-1,)): Bending stiffness of elements.
        l0 (ndarray of shape (N-1,)): Initial length of elements.
        dt (float): Time step size.
        damping (float): Nodal damping coefficient.

    Returns:
        x_next (ndarray of shape (N, 2)): Nodal positions at time t+dt.
        v_next (ndarray of shape (N, 2)): Nodal velocities at time t+dt.
        theta_next (ndarray of shape (N,)): Nodal rotations at time t+dt.
        omega_next (ndarray of shape (N,)): Nodal angular velocities at time t+dt.
    """

    # Elements vector and length
    dx = x[1:] - x[:-1]
    L = jnp.linalg.norm(dx, axis=-1)

    # Rigid body rotation of the element
    alpha = jnp.arctan2(dx[:, 1], dx[:, 0])

    # Pure deformations
    delta_d = L - l0
    theta_1d = theta[:-1] - alpha
    theta_2d = theta[1:] - alpha

    # Ensure angles are in [-pi, pi]
    theta_1d = jnp.mod(theta_1d + jnp.pi, 2 * jnp.pi) - jnp.pi
    theta_2d = jnp.mod(theta_2d + jnp.pi, 2 * jnp.pi) - jnp.pi

    # Internal forces in local frame
    fx = EA * delta_d / l0
    m1 = (EI / l0) * (4 * theta_1d + 2 * theta_2d)
    m2 = (EI / l0) * (2 * theta_1d + 4 * theta_2d)
    fy = (m1 + m2) / L

    # Transform internal forces to global frame
    cos_a = dx[:, 0] / L
    sin_a = dx[:, 1] / L

    f1x = -fx * cos_a - fy * sin_a
    f1y = -fx * sin_a + fy * cos_a
    f2x = fx * cos_a + fy * sin_a
    f2y = fx * sin_a - fy * cos_a

    # Assemble global internal forces on nodes
    f_int_x = jnp.pad(f1x, (0, 1)) + jnp.pad(f2x, (1, 0))
    f_int_y = jnp.pad(f1y, (0, 1)) + jnp.pad(f2y, (1, 0))
    m_int = jnp.pad(m1, (0, 1)) + jnp.pad(m2, (1, 0))

    f_int = jnp.stack([f_int_x, f_int_y], axis=-1)

    # Equations of motion
    a = (f_ext - f_int - damping * v) / m[:, None]
    alpha_acc = (m_ext - m_int - damping * omega) / J

    # Update velocities and positions (Semi-implicit Euler)
    v_next = v + a * dt
    omega_next = omega + alpha_acc * dt

    x_next = x + v_next * dt
    theta_next = theta + omega_next * dt

    return x_next, v_next, theta_next, omega_next
