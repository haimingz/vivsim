import jax.numpy as jnp

def newmark_2dof(a, v, d, h, m, k, c, dt=1, gamma=0.5, beta=0.25):
    """Newmark-beta method for dynamics
    
    Args: 
        a (float): acceleration at time t
        v (float): velocity at time t
        d (float): displacement at time t
        h (float): external force at time t
        m (float): mass
        k (float): stiffness
        c (float): damping
        dt (float): time step
        gamma (float): gamma parameter for Newmark-beta method
        beta (float): beta parameter for Newmark-beta method
    
    Returns:
        a_next (float): acceleration at time t+1
        v_next (float): velocity at time t+1
        d_next (float): displacement at time t+1
    """

    c1, c2 = gamma * dt, beta * dt ** 2
    v1 = v + dt * (1 - gamma) * a
    v2 = d + dt * v + dt**2 * (0.5 - beta) * a
    a_next = (h - c * v1 - k * v2) / (m + c1 * c + c2 * k)
    v_next = c1 * a_next + v1
    d_next = c2 * a_next + v2
    return a_next, v_next, d_next


def newmark_3dof(a, v, d, h, m, k, c, dt=1, gamma=0.5, beta=0.25):
    """Newmark-beta method for dynamics
    
    Args:
        a (matrix): acceleration at time t
        v (matrix): velocity at time t
        d (matrix): displacement at time t
        h (matrix): external force at time t
        m (matrix): mass
        k (matrix): stiffness
        c (matrix): damping
        dt (matrix): time step
        gamma (matrix): gamma parameter for Newmark-beta method
        beta (matrix): beta parameter for Newmark-beta method
    
    Returns:
        a_next (matrix): acceleration at time t+1
        v_next (matrix): velocity at time t+1
        d_next (matrix): displacement at time t+1
    """

    c1, c2 = gamma * dt, beta * dt ** 2

    v1 = v + dt * (1 - gamma) * a
    v2 = d + dt * v + dt ** 2 * (0.5 - beta) * a

    a = jnp.dot(jnp.linalg.inv(m + c1 * c + c2 * k),
               h - jnp.dot(c, v1) - jnp.dot(k, v2))

    a_next = a
    v_next = c1 * a_next + v1
    d_next = c2 * a_next + v2
    
    return a_next, v_next, d_next