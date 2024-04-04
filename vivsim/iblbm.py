import jax.numpy as jnp


def equilibrum(rho, u, feq):
    """
    Calculate the equilibrium distribution function for Lattice Boltzmann Method (LBM).

    Parameters:
    - rho (float): The density with shape (NX, NY).
    - u (list): The velocity vector with shape (2, NX, NY).
    - feq (list): The equilibrium distribution function with shape (9, NX, NY).

    Returns:
    - feq (list): The updated equilibrium distribution function with shape (9, NX, NY).
    """
    uxx = u[0] * u[0]
    uyy = u[1] * u[1]
    uxy = u[0] * u[1]
    uu = uxx + uyy
    
    feq = feq.at[0].set(4 / 9 * rho * (1 - 1.5 * uu))
    feq = feq.at[1].set(1 / 9 * rho * (1 - 1.5 * uu + 3 * u[0] + 4.5 * uxx))
    feq = feq.at[2].set(1 / 9 * rho * (1 - 1.5 * uu + 3 * u[1] + 4.5 * uyy))
    feq = feq.at[3].set(1 / 9 * rho * (1 - 1.5 * uu - 3 * u[0] + 4.5 * uxx))
    feq = feq.at[4].set(1 / 9 * rho * (1 - 1.5 * uu - 3 * u[1] + 4.5 * uyy))
    feq = feq.at[5].set(1 / 36 * rho * (1 + 3 * uu + 3 * (u[0] + u[1]) + 9 * uxy))
    feq = feq.at[6].set(1 / 36 * rho * (1 + 3 * uu - 3 * (u[0] - u[1]) - 9 * uxy))
    feq = feq.at[7].set(1 / 36 * rho * (1 + 3 * uu - 3 * (u[0] + u[1]) + 9 * uxy))
    feq = feq.at[8].set(1 / 36 * rho * (1 + 3 * uu + 3 * (u[0] - u[1]) - 9 * uxy))
    return feq


def streaming(f):
    """
    Performs the streaming step in the lattice Boltzmann method (LBM).

    Args:
    - f (ndarray): The distribution functions.

    Returns:
    - ndarray: The updated distribution functions after the streaming step.
    """
    f = f.at[1].set(jnp.roll(f[1], 1, axis=0))
    f = f.at[2].set(jnp.roll(f[2], 1, axis=1))
    f = f.at[3].set(jnp.roll(f[3], -1, axis=0))
    f = f.at[4].set(jnp.roll(f[4], -1, axis=1))
    f = f.at[5].set(jnp.roll(f[5], 1, axis=0))
    f = f.at[5].set(jnp.roll(f[5], 1, axis=1))
    f = f.at[6].set(jnp.roll(f[6], -1, axis=0))
    f = f.at[6].set(jnp.roll(f[6], 1, axis=1))
    f = f.at[7].set(jnp.roll(f[7], -1, axis=0))
    f = f.at[7].set(jnp.roll(f[7], -1, axis=1))
    f = f.at[8].set(jnp.roll(f[8], 1, axis=0))
    f = f.at[8].set(jnp.roll(f[8], -1, axis=1))
    return f


def get_macroscopic(f, rho, u):
    """
    Compute the density from the distribution functions.

    Args:
    - f (ndarray): The distribution functions.

    Returns:
    - ndarray: The density.
    """
    rho = jnp.sum(f, axis=0)
    u = u.at[0].set((f[1] + f[5] + f[8] - f[3] - f[6] - f[7]) / rho)
    u = u.at[1].set((f[2] + f[5] + f[6] - f[4] - f[7] - f[8]) / rho)
    return rho, u


def collision_bgk(f, feq, omega):
    """
    Perform the BGK collision step in the lattice Boltzmann method.

    Parameters:
    - f (ndarray): The distribution function with shape (9, NX, NY).
    - feq (ndarray): The equilibrium distribution function with shape (9, NX, NY).
    - omega (float): The relaxation parameter.

    Returns:
    - ndarray: The updated distribution function after the collision step.
    """
    return (1 - omega) * f + omega * feq


def get_omega_mrt(omega):
    """
    Generates the MRT collision matrix for a given value of omega.

    Parameters:
    - omega: A scalar value representing the relaxation parameter.

    Returns:
    - The MRT collision matrix.
    """
    # transformation matrix
    M = jnp.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [-4, -1, -1, -1, -1, 2, 2, 2, 2],
            [4, -2, -2, -2, -2, 1, 1, 1, 1],
            [0, 1, 0, -1, 0, 1, -1, -1, 1],
            [0, -2, 0, 2, 0, 1, -1, -1, 1],
            [0, 0, 1, 0, -1, 1, 1, -1, -1],
            [0, 0, -2, 0, 2, 1, 1, -1, -1],
            [0, 1, -1, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, 1, -1],
        ]
    )
    # relaxation matrix
    S = jnp.diag(jnp.array([1, 1.4, 1.4, 1, 1.2, 1, 1.2, omega, omega]))

    # MRT collision matrix
    return jnp.linalg.inv(M) @ S @ M


def collision_mrt(f, feq, omega_mrt):
    """
    Perform collision step using the Multiple Relaxation Time (MRT) model.

    Args:
        f (ndarray): The distribution function with shape (9, NX, NY).
        feq (ndarray): The equilibrium distribution function with shape (9, NX, NY).
        omega_mrt (ndarray): The relaxation rates for the MRT model with shape (9,9).

    Returns:
        ndarray: The updated distribution function after the collision step.
    """
    return jnp.tensordot(omega_mrt, feq - f, axes=([1], [0])) + f


def left_inlet(f, rho, u, u0):
    """
    Applies left inlet boundary conditions to the lattice Boltzmann method (LBM) simulation.

    Parameters:
    - f (ndarray): Distribution functions representing the particle populations with shape (9, NX, NY).
    - rho (ndarray): Density of the fluid with shape (NX, NY).
    - u (ndarray): Velocity of the fluid with shape (2, NX, NY).
    - u_inlet (float): Inlet velocity.

    Returns:
    - f (ndarray): Updated distribution functions after applying the boundary conditions.
    - rho (ndarray): Updated density after applying the boundary conditions.
    - u (ndarray): Updated velocity after applying the boundary conditions.
    """

    rho_wall = (f[0, 0] + f[2, 0] + f[4, 0] + 2 * (f[3, 0] + f[6, 0] + f[7, 0])) / (
        1 - u0
    )
    rho = rho.at[0].set(rho_wall)
    u = u.at[0, 0].set(u0)
    u = u.at[1, 0].set(0)
    f = f.at[1, 0].set(f[3, 0] + 2 / 3 * rho_wall * u0)
    f = f.at[5, 0].set(f[7, 0] - 0.5 * (f[2, 0] - f[4, 0]) + 1 / 6 * rho_wall * u0)
    f = f.at[8, 0].set(f[6, 0] + 0.5 * (f[2, 0] - f[4, 0]) + 1 / 6 * rho_wall * u0)
    return f, rho, u


def right_outlet(f):
    """
    Outlet BC at right wall (No gradient BC)

    Args:
    - f: Distribution functions

    Returns:
    - f: Updated distribution functions
    """

    f = f.at[3, -1].set(f[3, -2])
    f = f.at[6, -1].set(f[6, -2])
    f = f.at[7, -1].set(f[7, -2])
    return f


# kernel functions
def stencil2(distance):
    return jnp.where(jnp.abs(distance) <= 1, 1 - jnp.abs(distance), 0)


def stencil3(distance):
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


def kernel3(x, y, X, Y):
    return stencil3(X - x) * stencil3(Y - y)


def kernel2(x, y, X, Y):
    return stencil2(X - x) * stencil2(Y - y)


def intepolate(u, kernel):
    return jnp.einsum("xy,dxy->d", kernel, u)


def spreading(g, kernel):
    return kernel * g[:, None, None]


def get_correction_g(v, u):
    return 2 * (v - u)

def get_correction_u(g):
    return g * 0.5


def get_correction_f(u, g, omega):
    """fit fluid force to the lattice"""

    gxux = g[0] * u[0]
    gyuy = g[1] * u[1]
    gxuy = g[0] * u[1]
    gyux = g[1] * u[0]
    c = (1 - 0.5 * omega)

    _ = jnp.zeros((9, u.shape[1], u.shape[2]))
    
    _ = _.at[0].set(4 / 3 * (-gxux - gyuy)) 
    _ = _.at[1].set(1 / 3 * (2 * gxux + g[0] - gyuy)) 
    _ = _.at[2].set(1 / 3 * (2 * gyuy + g[1] - gxux)) 
    _ = _.at[3].set(1 / 3 * (2 * gxux - g[0] - gyuy)) 
    _ = _.at[4].set(1 / 3 * (2 * gyuy - g[1] - gxux)) 
    _ = _.at[5].set(1 / 12 * (2 * gxux + 3 * gxuy + g[0] + 3 * gyux + 2 * gyuy + g[1])) 
    _ = _.at[6].set(1 / 12 * (2 * gxux - 3 * gxuy - g[0] - 3 * gyux + 2 * gyuy + g[1])) 
    _ = _.at[7].set(1 / 12 * (2 * gxux + 3 * gxuy - g[0] + 3 * gyux + 2 * gyuy - g[1])) 
    _ = _.at[8].set(1 / 12 * (2 * gxux - 3 * gxuy + g[0] - 3 * gyux + 2 * gyuy - g[1])) 

    return _ * c

def newmark(a, v, d, g, m, k, c, dt=1, gamma=0.5, beta=0.25):
    """Newmark-beta method for dynamics"""

    c1, c2 = gamma * dt, beta * dt**2
    v1 = v + dt * (1 - gamma) * a
    v2 = d + dt * v + dt**2 * (0.5 - beta) * a
    a_next = (g - c * v1 - k * v2) / (m + c1 * c + c2 * k)
    v_next = c1 * a_next + v1
    d_next = c2 * a_next + v2
    return a_next, v_next, d_next
