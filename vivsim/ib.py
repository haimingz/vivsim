"""
This module implements the Immersed Boundary Method (IBM) for the LBM framework.

The IBM enables fluid-structure interaction by transferring forces between the fluid
and solid objects. The fluid-solid interface is represented by a set of Lagrangian
markers that can move freely through the fixed Eulerian grid.

Two main IBM approaches are implemented:
    * Multi-Direct Forcing: Iterative method for enforcing no-slip boundary conditions
    * Implicit Method: Direct solution using matrix inversion

Key Variables:
    * u: Fluid velocity field, shape (2, NX, NY)
    * u_markers: Interpolated fluid velocity at Lagrangian markers, shape (N_MARKER, 2)
    * v_markers: Velocity of Lagrangian markers, shape (N_MARKER, 2)
    * g: Force density that object applies to fluid, shape (2, NX, NY)
    * g_markers: Force at Lagrangian markers applied to fluid, shape (N_MARKER, 2)
    * h_markers: Force at Lagrangian markers applied to solid, shape (N_MARKER, 2)
    * kernels: Interpolation/spreading kernel functions, shape (N_MARKER, NX, NY)
    * ds_markers: Differential segment length of markers, scalar or shape (N_MARKER,)

where NX, NY are the grid dimensions and N_MARKER is the number of Lagrangian markers.
"""

import jax
import jax.numpy as jnp
from .lbm import get_velocity_correction

# ----------------- Kernel functions -----------------


def kernel_range2(distance):
    """Kernel function of range 2
    
    Args:
        distance (scalar or jax.Array): The distance between the marker and the lattice node.
    
    Returns:
        out (scalar or jax.Array): The kernel function value in range [0, 1].
    """
    
    return jnp.where(jnp.abs(distance) <= 1, 1 - jnp.abs(distance), 0)


def kernel_range3(distance):
    """Kernel function of range 3
    
    Args:
        distance (scalar or jax.Array): The distance between the marker and the lattice node.
    
    Returns:
        out (scalar or jax.Array): The kernel function value in range [0, 1/3].
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
    """Kernel function of range 4.
    
    Args:
        distance (scalar or jax.Array): The distance between the marker and the lattice node.
    
    Returns:
        out (scalar or jax.Array): The kernel function value in range [0, 1/8].
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


def get_kernels(x_markers, y_markers, x_grid, y_grid, kernel_func):
    """Generate interpolation/spreading kernels for all Lagrangian markers.
    
    Precomputes the kernel functions for efficient interpolation (Eulerian to Lagrangian)
    and spreading (Lagrangian to Eulerian) operations. The kernel is the tensor product
    of 1D kernel functions in x and y directions.
    
    Args:
        x_markers (jax.Array of shape (N_MARKER,)): x-coordinates of Lagrangian markers.
        y_markers (jax.Array of shape (N_MARKER,)): y-coordinates of Lagrangian markers.
        x_grid (jax.Array of shape (NX, NY)): x-coordinates of Eulerian grid nodes.
        y_grid (jax.Array of shape (NX, NY)): y-coordinates of Eulerian grid nodes.
        kernel_func (callable): The kernel function. Options: 
            kernel_range2, kernel_range3, kernel_range4.
    
    Returns:
        kernels (jax.Array of shape (N_MARKER, NX, NY)): Stacked kernel functions
            for all markers. Each slice kernels[i, :, :] gives the kernel weights
            for marker i over all grid points.
    """
    return (kernel_func(x_grid[None, ...] - x_markers[:, None, None]) \
          * kernel_func(y_grid[None, ...] - y_markers[:, None, None]))


def get_area(coord_markers):
    """
    Calculate the area of a closed polygon using shoelace formula.
    The polygon is defined by its vertices in the order they are connected.

    Parameters:
    - coord_markers: A JAX array of shape (N, 2) representing the (x, y) coordinates of the polygon vertices.

    Returns:
    - area: The area of the polygon (a JAX scalar).
    """
    x = coord_markers[:, 0]
    y = coord_markers[:, 1]
    area = 0.5 * jnp.abs(jnp.sum(x * jnp.roll(y, 1) - y * jnp.roll(x, 1)))
    return area


def get_ds_closed(coord_markers):
    """
    Calculate the differential segment length (ds) for each point on a closed curve.
    This represents the segment length element associated with each point for integration.
    
    For each point, ds is the average of the two adjacent segment lengths.

    Parameters:
    - coord_markers: A numpy array of shape (N, 2) representing the (x, y) coordinates of the curve points.

    Returns:
    - ds: A numpy array of shape (N,) containing the differential segment length for each point.
    """
    # Calculate segment lengths (distance from each point to the next)
    segment_lengths = jnp.linalg.norm(coord_markers - jnp.roll(coord_markers, shift=-1, axis=0), axis=1)
    
    # For each point, average the segment before and after it
    ds = (segment_lengths + jnp.roll(segment_lengths, shift=1)) / 2.0
    return ds


def get_ds_open(coord_markers):
    """
    Calculate the differential segment length (ds) for each point on an open curve.
    This represents the segment length element associated with each point for integration.
    
    For interior points, ds is the average of the two adjacent segment lengths.
    For the first and last points, ds is half the length of their adjacent segment.

    Parameters:
    - coord_markers: A numpy array of shape (N, 2) representing the (x, y) coordinates of the curve points.

    Returns:
    - ds: A numpy array of shape (N,) containing the differential segment length for each point.
    """
    # Calculate segment lengths
    segment_lengths = jnp.linalg.norm(coord_markers[1:] - coord_markers[:-1], axis=1)
    
    # Vectorized calculation: each point gets half of adjacent segments
    ds = jnp.zeros(len(coord_markers))
    ds[:-1] += segment_lengths / 2  # Add half to start of each segment
    ds[1:] += segment_lengths / 2    # Add half to end of each segment
    
    return ds




# ----------------- Core IB calculation -----------------


def multi_direct_forcing(u, v_markers, kernels, ds_markers, n_iter=5):
    """Multi-Direct Forcing (Iterative) IBM for moving boundaries.
    
    Args:
        u (jax.Array of shape (2, NX, NY)): Fluid velocity field.
        v_markers (jax.Array of shape (N_MARKER, 2)): Velocity of Lagrangian markers.
        kernels (jax.Array of shape (N_MARKER, NX, NY)): Kernel functions
            from get_kernels(). This should be recomputed when markers move. 
        ds_markers (scalar or jax.Array of shape (N_MARKER,)): Differential segment length 
            associated with each marker (for force integration).
        n_iter (int, optional): Number of iterations. Default is 5.
    
    Returns:
        g (jax.Array of shape (2, NX, NY)): Force density field applied to fluid.
        h_markers (jax.Array of shape (N_MARKER, 2)): Forces applied to individual markers
            (equal and opposite to the force applied to fluid).
    """
    
    ds_markers = jnp.reshape(ds_markers, (-1, 1))  # Ensure correct shape for broadcasting

    g = jnp.zeros_like(u)
    h_markers = jnp.zeros((kernels.shape[0], 2))
    
    def loop_body(i, carry):
        g, h_markers, u = carry
        
        # Interpolate velocity to markers
        u_markers = jnp.einsum("dxy,nxy->nd", u, kernels)
        
        # Compute force correction at markers
        g_markers_correction = (v_markers - u_markers) * ds_markers * 2
        
        # Spread force to Eulerian grid
        g_correction = jnp.einsum("nd,nxy->dxy", g_markers_correction, kernels)
        g += g_correction
        
        # Accumulate force on solid (action-reaction)
        h_markers -= g_markers_correction
        
        # Update velocity field
        u += get_velocity_correction(g_correction) 
        
        return g, h_markers, u
        
    g, h_markers, _ = jax.lax.fori_loop(0, n_iter, loop_body, (g, h_markers, u))
    
    return g, h_markers

