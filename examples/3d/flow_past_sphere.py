r"""
Flow past a stationary sphere using 3D IB-LBM

This example simulates incompressible flow past a fixed sphere using the
Immersed Boundary Method (IBM) coupled with a D3Q19 Lattice Boltzmann solver.
At moderate Reynolds numbers the wake develops vortex shedding with hairpin
structures that are visualised with a live mid-plane vorticity slice.

"""

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from tqdm import tqdm

from vivsim import ib3d, lbm3d, post


# ========================== VISUALIZATION PARAMETERS ====================

PLOT = True           # Enable live pyvista 3D rendering
CHUNK_STEPS = 50      # Simulation steps per compute chunk


# ========================== GEOMETRY =======================

D = 24          # Sphere diameter
NX = 12 * D     # Domain length (x)
NY = 4 * D      # Domain width (y)
NZ = 4 * D      # Domain height (z)
SPH_X = 4 * D   # Sphere center x-position
SPH_Y = NY // 2  # Sphere center y-position
SPH_Z = NZ // 2  # Sphere center z-position
SLICE_Z = SPH_Z

# Generate icosphere surface mesh for the IB markers


def _icosphere(radius, center, subdivisions=2):
    """Generate an icosphere triangulation (vertices and faces)."""
    # Golden ratio
    phi = (1 + math.sqrt(5)) / 2

    # 12 vertices of an icosahedron
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts[0])

    # 20 triangular faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)

    # Subdivide
    for _ in range(subdivisions):
        edge_midpoint = {}
        new_faces = []
        for tri in faces:
            mids = []
            for i in range(3):
                edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
                if edge not in edge_midpoint:
                    mid = (verts[edge[0]] + verts[edge[1]]) / 2
                    mid /= np.linalg.norm(mid)
                    edge_midpoint[edge] = len(verts)
                    verts = np.vstack([verts, mid])
                mids.append(edge_midpoint[edge])
            a, b, c = tri
            m0, m1, m2 = mids
            new_faces.extend([
                [a, m0, m2], [b, m1, m0], [c, m2, m1], [m0, m1, m2],
            ])
        faces = np.array(new_faces, dtype=np.int64)

    # Scale and translate
    verts = verts * radius + np.array(center)
    return verts, faces


MARKER_COORDS_NP, MARKER_FACES = _icosphere(D / 2, [SPH_X, SPH_Y, SPH_Z], subdivisions=4)
MARKER_COORDS = jnp.array(MARKER_COORDS_NP, dtype=jnp.float32)
N_MARKER = MARKER_COORDS.shape[0]
MARKER_DS = ib3d.get_vertex_dA(MARKER_COORDS, jnp.array(MARKER_FACES))  # Surface area weights


# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.05      # Inlet velocity
RE = 2000       # Reynolds number

# Derived physical parameters
NU = U0 * D / RE  # Kinematic viscosity
TM = int(40 * D / U0)  # Total simulation timesteps


# ========================== IB-LBM PARAMETERS ==========================

OMEGA = lbm3d.get_omega(NU)  # Relaxation parameter

IB_ITER = 3    # Multi-direct forcing iterations for IBM coupling
IB_PAD = 4     # Padding around sphere for defining the local IBM region

IB_X0 = int(SPH_X - D / 2 - IB_PAD)
IB_Y0 = int(SPH_Y - D / 2 - IB_PAD)
IB_Z0 = int(SPH_Z - D / 2 - IB_PAD)
IB_SIZE = D + 2 * IB_PAD  # Size of the cubic IBM region

# Pre-compute marker stencils in the local IBM region
MARKER_COORDS_IB = MARKER_COORDS - jnp.array([IB_X0, IB_Y0, IB_Z0])
STENCIL_WEIGHTS, STENCIL_INDICES = ib3d.get_ib_stencil(
    MARKER_COORDS_IB,
    grid_shape=(IB_SIZE, IB_SIZE, IB_SIZE),
    kernel=ib3d.kernel_peskin_4pt,
)


# ======================= INITIALIZE VARIABLES ====================

rho = jnp.ones((NX, NY, NZ))              # Fluid density
u = jnp.zeros((3, NX, NY, NZ))            # Fluid velocity
u = u.at[0].set(U0)                       # Set initial x-velocity to U0
f = lbm3d.get_equilibrium(rho, u)         # Initial distribution function

marker_v = jnp.zeros((N_MARKER, 3))       # Target marker velocity (zero for fixed sphere)


# ======================= SIMULATION ROUTINE =====================

def update_step(f):

    # Collision
    rho, u = lbm3d.get_macroscopic(f)
    feq = lbm3d.get_equilibrium(rho, u)
    f = lbm3d.collision_kbc(f, feq, OMEGA)

    # Extract IBM region data for efficient computation
    ib_rho = jax.lax.dynamic_slice(rho, (IB_X0, IB_Y0, IB_Z0),
                                   (IB_SIZE, IB_SIZE, IB_SIZE))
    ib_u = jax.lax.dynamic_slice(u, (0, IB_X0, IB_Y0, IB_Z0),
                                 (3, IB_SIZE, IB_SIZE, IB_SIZE))
    ib_f = jax.lax.dynamic_slice(f, (0, IB_X0, IB_Y0, IB_Z0),
                                 (19, IB_SIZE, IB_SIZE, IB_SIZE))

    # Run multi-direct forcing to enforce no-slip condition on the sphere
    ib_g, _ = ib3d.multi_direct_forcing(
        grid_u=ib_u,
        stencil_weights=STENCIL_WEIGHTS,
        stencil_indices=STENCIL_INDICES,
        marker_u_target=marker_v,
        marker_ds=MARKER_DS,
        n_iter=IB_ITER,
    )

    # Apply IBM forcing to the fluid in the local IBM region
    ib_f = lbm3d.forcing_edm(ib_f, ib_g, ib_u, ib_rho)
    f = jax.lax.dynamic_update_slice(f, ib_f, (0, IB_X0, IB_Y0, IB_Z0))

    # Streaming and boundary conditions
    f = lbm3d.streaming(f)
    f = lbm3d.boundary_nebb(f, loc="left", ux_wall=U0)
    f = lbm3d.boundary_equilibrium(f, loc="right", ux_wall=U0)

    return f


@partial(jax.jit, static_argnums=1, donate_argnums=0)
def update_chunk(carry, n_steps):
    def step(carry, _):
        (f,) = carry
        f = update_step(f)
        return (f,), None
    return jax.lax.scan(step, carry, None, length=n_steps)





def build_midplane_slice(u):
    """Build the mid-span vorticity slice."""
    vort_mag = np.asarray(post.vorticity_magnitude(u))
    slice_grid = pv.ImageData(
        dimensions=(NX, NY, 1),
        origin=(0.0, 0.0, float(SLICE_Z)),
        spacing=(1.0, 1.0, 1.0),
    )
    slice_grid.point_data["vorticity_magnitude"] = vort_mag[:, :, SLICE_Z].ravel(order="F")
    slice_grid.set_active_scalars("vorticity_magnitude")
    return slice_grid


def get_color_limits(values, previous_limits=None):
    """Return a stable scalar range for the vorticity slice."""
    vmax = float(np.percentile(values, 99))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    vmax *= 1.05

    if previous_limits is None:
        return (0.0, vmax)

    alpha = 0.2
    smoothed_max = (1.0 - alpha) * previous_limits[1] + alpha * vmax
    return (0.0, smoothed_max)


# ========================== RUN SIMULATION ==========================

chunk_sizes = [CHUNK_STEPS] * (TM // CHUNK_STEPS)
if TM % CHUNK_STEPS:
    chunk_sizes.append(TM % CHUNK_STEPS)

pl = None
slice_actor = None
current_step = 0
camera_framed = False
color_limits = None

if PLOT:
    pl = pv.Plotter()
    pl.set_background("white")
    pl.camera_position = [
        (1.8 * NX, 1.35 * NY, 1.25 * NZ),
        (0.55 * NX, 0.5 * NY, 0.5 * NZ),
        (0.0, 0.0, 1.0),
    ]
    domain_box = pv.Box(bounds=(0.0, NX - 1.0, 0.0, NY - 1.0, 0.0, NZ - 1.0))
    sphere = pv.Sphere(
        radius=D / 2,
        center=(SPH_X, SPH_Y, SPH_Z),
        theta_resolution=40,
        phi_resolution=40,
    )
    initial_slice = build_midplane_slice(u)
    color_limits = get_color_limits(initial_slice["vorticity_magnitude"])
    pl.add_mesh(domain_box, style="wireframe", color="black", line_width=1.0, opacity=0.35)
    pl.add_mesh(sphere, color="lightgray", smooth_shading=True, specular=0.1)
    slice_actor = pl.add_mesh(
        initial_slice,
        scalars="vorticity_magnitude",
        cmap="inferno",
        clim=color_limits,
        opacity=0.95,
        show_scalar_bar=False,
        lighting=False,
    )
    pl.show(interactive_update=True, auto_close=False)

with tqdm(total=TM, unit="step") as pbar:
    for n_steps in chunk_sizes:
        (f,), _ = update_chunk((f,), n_steps)
        jax.block_until_ready(f)
        current_step += n_steps
        pbar.update(n_steps)

        if PLOT:
            _, u_now = lbm3d.get_macroscopic(f)
            slice_grid = build_midplane_slice(u_now)
            color_limits = get_color_limits(slice_grid["vorticity_magnitude"], color_limits)
            slice_actor.mapper.dataset.copy_from(slice_grid)
            slice_actor.mapper.scalar_range = color_limits
            if not camera_framed:
                pl.reset_camera()
                camera_framed = True
            pl.render()
            pl.update()


if PLOT and pl is not None:
    pl.show()
