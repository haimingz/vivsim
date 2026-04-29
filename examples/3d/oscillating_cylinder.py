r"""
Vortex-Induced Vibration (VIV) of a finite cylinder using 3D IB-LBM

This example simulates incomprehensible fluid flow past a finite cylinder that 
undergoes Vortex-Induced Vibration (VIV). It demonstrates the fluid-structure
interaction (FSI) capabilities of the Immersed Boundary Method (IBM) coupled 
with a rigid body dynamics solver.

We use PyVista to visualize the Q-criterion isosurfaces and a midplane 
Z-vorticity slice to reveal the beautiful vortex wake and flow development.
"""

from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from tqdm import tqdm

from vivsim import dyn, ib3d, lbm3d, post

# Fix for PyVista empty mesh plotting issue
pv.global_theme.allow_empty_mesh = True

# ========================== VISUALIZATION PARAMETERS ====================

PLOT = True           # Enable live pyvista 3D rendering
CHUNK_STEPS = 100     # Simulation steps per compute chunk

# Q-criterion threshold: higher = thinner vortex tubes (only show strong cores)
# Raise to 1e-4 for very thin tubes; lower to 1e-5 for thicker/more tubes.
# Rule of thumb: Q ~ (ω_z)^2 / 4, so Q=5e-5 shows regions where |ω_z| > 0.014
Q_THRESHOLD = 5e-5


# ========================== GEOMETRY =======================

D = 24          # Cylinder diameter
NX = 10 * D        # Domain length (x)
NY = 6 * D         # Domain width (y)
NZ = 5 * D         # Domain height (z)
CYL_X = 3 * D   # Cylinder mean center x-position
CYL_Y = NY // 2 # Cylinder mean center y-position
CYL_Z = NZ // 2 # Cylinder mean center z-position
CYL_H = NZ - 24 # Cylinder length (finite, avoids z-boundaries)

def _make_uniform_cylinder_mesh(center, radius, height, theta_res, z_res):
    """Build a uniform triangulated cylinder mesh with explicit z-layers.

    pv.Cylinder() only has 2 z-layers (top + bottom) on the lateral surface.
    After subdivide(3) the spanwise spacing remains ~7 lu (need <= 0.5 lu),
    causing fluid to leak through the cylinder and destroying the wake.
    This function generates a proper uniform mesh with:
        circumferential spacing = pi*D / theta_res  (target: 0.5 lu)
        spanwise spacing        = height / z_res    (target: 0.5 lu)
    End caps are triangulated with concentric rings (spacing ~0.5 lu) to
    avoid the dA singularity that occurs with a single fan-center vertex.
    """
    cx, cy, cz = center
    z_bot = cz - height / 2.0
    z_top = cz + height / 2.0

    theta = np.linspace(0.0, 2.0 * math.pi, theta_res, endpoint=False)
    z_vals = np.linspace(z_bot, z_top, z_res + 1)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Lateral surface vertices: shape (z_res+1, theta_res, 3)
    lat_x = (cx + radius * cos_t)[None, :] * np.ones((z_res + 1, 1))
    lat_y = (cy + radius * sin_t)[None, :] * np.ones((z_res + 1, 1))
    lat_z = z_vals[:, None] * np.ones((1, theta_res))
    lat_pts = np.stack([lat_x, lat_y, lat_z], axis=-1).reshape(-1, 3)

    # Lateral triangles: each (i,j) quad -> 2 triangles
    lat_faces = []
    for i in range(z_res):
        for j in range(theta_res):
            j1 = (j + 1) % theta_res
            v00 = i * theta_res + j
            v01 = i * theta_res + j1
            v10 = (i + 1) * theta_res + j
            v11 = (i + 1) * theta_res + j1
            # Outward normals require CCW winding viewed from outside:
            # [v00, v11, v10] and [v00, v01, v11]
            lat_faces.extend([[v00, v11, v10], [v00, v01, v11]])

    n_lat = len(lat_pts)

    def _cap_mesh(z_cap, outward_down):
        """Concentric-ring cap: ring spacing ~0.5 lu, avoids a dA-spike center vertex."""
        cap_pts = []
        cap_faces = []
        # Number of rings: ceil(radius / 0.5), innermost ring at r_step
        r_step = 0.5
        n_rings = max(1, int(math.ceil(radius / r_step)))
        r_values = np.linspace(0.0, radius, n_rings + 1)  # includes r=0 and r=radius

        ring_offsets = []  # start index of each ring in cap_pts
        for ri, r in enumerate(r_values):
            if r == 0.0:
                # Single center point
                ring_offsets.append(len(cap_pts))
                cap_pts.append([cx, cy, z_cap])
            else:
                # Number of points on this ring proportional to circumference, min theta_res
                n_pts = max(theta_res, int(math.ceil(2 * math.pi * r / r_step)))
                t = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
                ring_offsets.append(len(cap_pts))
                for ti in t:
                    cap_pts.append([cx + r * math.cos(ti), cy + r * math.sin(ti), z_cap])

        # Triangulate between consecutive rings
        for ri in range(len(r_values) - 1):
            o_inner = ring_offsets[ri]
            o_outer = ring_offsets[ri + 1]
            n_inner = 1 if r_values[ri] == 0.0 else (ring_offsets[ri + 1] - o_inner)
            n_outer = ring_offsets[ri + 2] - o_outer if ri + 2 < len(ring_offsets) else (len(cap_pts) - o_outer)

            if n_inner == 1:
                # Fan from center to first ring
                for j in range(n_outer):
                    j1 = (j + 1) % n_outer
                    tri = [o_inner, o_outer + j, o_outer + j1]
                    cap_faces.append(tri if outward_down else tri[::-1])
            else:
                # Quad-strip (advancing front) between two rings
                i_inner, i_outer = 0, 0
                while i_inner < n_inner or i_outer < n_outer:
                    vi0 = o_inner + i_inner % n_inner
                    vi1 = o_inner + (i_inner + 1) % n_inner
                    vo0 = o_outer + i_outer % n_outer
                    vo1 = o_outer + (i_outer + 1) % n_outer
                    angle_inner = 2 * math.pi * (i_inner + 1) / n_inner
                    angle_outer = 2 * math.pi * (i_outer + 1) / n_outer
                    if angle_inner <= angle_outer:
                        tri = [vi0, vo0, vi1]
                        i_inner += 1
                    else:
                        tri = [vi0, vo0, vo1]
                        i_outer += 1
                    cap_faces.append(tri if outward_down else tri[::-1])

        return np.array(cap_pts), np.array(cap_faces, dtype=np.int32)

    bot_pts, bot_faces_local = _cap_mesh(z_bot, outward_down=True)
    top_pts, top_faces_local = _cap_mesh(z_top, outward_down=False)

    n_bot = len(bot_pts)
    bot_faces = bot_faces_local + n_lat
    top_faces = top_faces_local + n_lat + n_bot

    all_pts = np.vstack([lat_pts, bot_pts, top_pts])
    all_faces = np.array(lat_faces + bot_faces.tolist() + top_faces.tolist(), dtype=np.int32)
    return all_pts, all_faces


# Circumferential spacing = pi*D / theta_res <= 0.5  ->  theta_res >= pi*D/0.5 ~ 100
# Spanwise spacing        = CYL_H / z_res   <= 0.5  ->  z_res >= CYL_H/0.5 = 112
THETA_RES = 100
Z_RES     = int(math.ceil(CYL_H / 0.5))   # = 112

MARKER_INITIAL_NP, MARKER_FACES = _make_uniform_cylinder_mesh(
    center=(CYL_X, CYL_Y, CYL_Z),
    radius=D / 2,
    height=CYL_H,
    theta_res=THETA_RES,
    z_res=Z_RES,
)

MARKER_INITIAL = jnp.array(MARKER_INITIAL_NP, dtype=jnp.float32)
N_MARKER = MARKER_INITIAL.shape[0]
MARKER_DS = ib3d.get_ds(MARKER_INITIAL, jnp.array(MARKER_FACES))


# ========================== PHYSICAL PARAMETERS =====================

U0 = 0.05      # Inlet velocity
RE = 1000       # Reynolds number based on diameter
NU = U0 * D / RE  # Kinematic viscosity

# VIV Structural Parameters
UR = 5         # Reduced velocity (U0 / (FN * D))
MR = 2        # Mass ratio (cylinder mass / displaced fluid mass)
DR = 0         # Damping ratio

FN = U0 / (UR * D)         # Natural frequency of the structure
CYL_VOL = math.pi * (D / 2)**2 * CYL_H
M = CYL_VOL * MR           # Mass of the cylinder
K = (2 * math.pi * FN) ** 2 * M * (1 + 1 / MR) # Spring stiffness
C = 2 * math.sqrt(K * M) * DR # Damping coefficient

TM = int(120 * D / U0)  # Total simulation timesteps


# ========================== IB-LBM PARAMETERS ==========================

OMEGA = lbm3d.get_omega(NU)

IB_ITER = 3    # Uniform mesh converges well with 3 iterations
IB_PAD = 6
# IB_SIZE_XY must cover the cylinder + VIV displacement headroom.
# At MR=2, Re=500, UR=5: peak amplitude A/D ~ 0.5-1.0 -> 8-16 lu displacement.
# Set window = D + 2*(IB_PAD + D) to safely hold ±D lateral excursion.
IB_SIZE_XY = int(D + 2 * (IB_PAD + D))   # = 60
IB_SIZE_Z = int(CYL_H + 2 * IB_PAD)


# ======================= INITIALIZE VARIABLES ====================

rho = jnp.ones((NX, NY, NZ))
u = jnp.zeros((3, NX, NY, NZ))
u = u.at[0].set(U0)
f = lbm3d.get_equilibrium(rho, u)

# Structural variables (X, Y)
d = jnp.zeros(2)            # Displacement of cylinder
v = jnp.zeros(2)            # Velocity of cylinder
a = jnp.zeros(2)            # Acceleration of cylinder

# Provide a small perturbation to kickstart the VIV instability earlier
v = v.at[1].set(1e-2 * U0)


# ======================= SIMULATION ROUTINE =====================

@partial(jax.jit, static_argnums=1, donate_argnums=0)
def update_chunk(carry, n_steps):
    def step(carry, _):
        f, d, v, a, step_idx = carry

        # Collision
        rho, u = lbm3d.get_macroscopic(f)
        feq = lbm3d.get_equilibrium(rho, u)
        f_post = lbm3d.collision_kbc(f, feq, OMEGA)

        # Dynamic slicing for the local IBM region
        center = jnp.array([CYL_X + d[0], CYL_Y + d[1], CYL_Z])
        marker_coords = MARKER_INITIAL + jnp.array([d[0], d[1], 0.])

        ib_x0 = jnp.clip(jnp.floor(center[0] - IB_SIZE_XY/2).astype(jnp.int32), 0, NX - IB_SIZE_XY)
        ib_y0 = jnp.clip(jnp.floor(center[1] - IB_SIZE_XY/2).astype(jnp.int32), 0, NY - IB_SIZE_XY)
        ib_z0 = jnp.clip(jnp.floor(center[2] - IB_SIZE_Z/2).astype(jnp.int32), 0, NZ - IB_SIZE_Z)

        ib_rho = jax.lax.dynamic_slice(rho, (ib_x0, ib_y0, ib_z0), (IB_SIZE_XY, IB_SIZE_XY, IB_SIZE_Z))
        ib_u = jax.lax.dynamic_slice(u, (0, ib_x0, ib_y0, ib_z0), (3, IB_SIZE_XY, IB_SIZE_XY, IB_SIZE_Z))
        ib_f = jax.lax.dynamic_slice(f_post, (0, ib_x0, ib_y0, ib_z0), (19, IB_SIZE_XY, IB_SIZE_XY, IB_SIZE_Z))

        marker_coords_ib = marker_coords - jnp.array([ib_x0, ib_y0, ib_z0])
        stencil_weights, stencil_indices = ib3d.get_ib_stencil(
            marker_coords_ib,
            grid_shape=(IB_SIZE_XY, IB_SIZE_XY, IB_SIZE_Z),
            kernel=ib3d.kernel_peskin_4pt,
        )

        a_old, v_old, d_old = a, v, d
        marker_v = jnp.zeros((N_MARKER, 3)).at[:, :2].set(v)

        ib_g, marker_h = ib3d.multi_direct_forcing(
            grid_u=ib_u,
            stencil_weights=stencil_weights,
            stencil_indices=stencil_indices,
            marker_u_target=marker_v,
            marker_ds=MARKER_DS,
            n_iter=IB_ITER,
        )

        # Update Structural Dynamics
        h = dyn.get_force_to_obj(marker_h)[:2]
        h += a * CYL_VOL  # Add displaced mass logic
        a, v, d = dyn.newmark_2dof(a_old, v_old, d_old, h, M, K, C)

        ib_f = lbm3d.forcing_edm(ib_f, ib_g, ib_u)
        f_post = jax.lax.dynamic_update_slice(f_post, ib_f, (0, ib_x0, ib_y0, ib_z0))

        # Streaming
        f_next = lbm3d.streaming(f_post)
        f_next = lbm3d.boundary_nebb(f_next, loc="left", ux_wall=U0)
        f_next = lbm3d.boundary_equilibrium(f_next, loc="right", ux_wall=U0)

        # Return carry + [dy, hy] for lift-force diagnostics
        return (f_next, d, v, a, step_idx + 1), jnp.stack([d[1], h[1]])
    return jax.lax.scan(step, carry, None, length=n_steps)


# ========================== VISUALIZATION UTILS ==========================

# Color Q-isosurface by spanwise vorticity ω_z (= ∂u_y/∂x - ∂u_x/∂y).
# For a Z-axis cylinder with Z-aligned Kármán vortex tubes, ω_z is dominant.
# Near-surface shear: ω_z ~ U0/δ_BL ~ U0*sqrt(Re)/D ~ 10*U0/D at Re=500.
# Wake vortex core: ω_z ~ 3-8 × U0/D.  Use ±10*U0/D as a safe headroom.
OMEGA_Z_CLIM = 10.0 * U0 / D   # = 0.03125


def get_q_criterion_isosurface(u, q_threshold=Q_THRESHOLD):
    """Extract Q-criterion isosurface colored by spanwise vorticity ω_z.

    The cylinder axis is Z, so the Kármán vortex tubes are Z-aligned and their
    dominant vorticity component is ω_z = ∂u_y/∂x − ∂u_x/∂y.
    Both Q and ω_z are embedded in the ImageData grid; PyVista interpolates
    ω_z onto the Q-isosurface automatically at contour-extraction time.
    """
    vort  = np.asarray(post.vorticity(u))
    omega_z = vort[2].copy()
    q = np.asarray(post.q_criterion(u)).copy()

    # Zero out boundary halos to suppress artifacts
    for arr in (q, omega_z):
        arr[:4, :, :] = 0;  arr[-4:, :, :] = 0
        arr[:, :4, :] = 0;  arr[:, -4:, :] = 0
        arr[:, :, :4] = 0;  arr[:, :, -4:] = 0

    grid = pv.ImageData(
        dimensions=(NX, NY, NZ),
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
    )
    grid.point_data["Q"]       = q.ravel(order="F")
    grid.point_data["omega_z"] = omega_z.ravel(order="F")

    try:
        isosurf = grid.contour([q_threshold], scalars="Q")
        if isosurf.n_points == 0:
            return pv.PolyData()
    except Exception:
        return pv.PolyData()

    # CRITICAL: contour(scalars="Q") leaves "Q" as the active scalar.
    # Q is *constant* on the isosurface (= q_threshold), so every vertex maps
    # to the same value → uniform white rendering.
    # We must switch the active scalar to omega_z BEFORE returning.
    isosurf.set_active_scalars("omega_z")
    return isosurf



# ========================== RUN SIMULATION ==========================

chunk_sizes = [CHUNK_STEPS] * (TM // CHUNK_STEPS)
if TM % CHUNK_STEPS:
    chunk_sizes.append(TM % CHUNK_STEPS)

if PLOT:
    pl = pv.Plotter()
    pl.set_background("black")
    
    pl.camera_position = [
        (NX * 0.8, -NY * 0.5, NZ * 1.5),
        (NX * 0.45, NY * 0.5, NZ * 0.5),
        (0, 0, 1)
    ]
    
    domain_box = pv.Box(bounds=(0.0, NX - 1.0, 0.0, NY - 1.0, 0.0, NZ - 1.0))
    pl.add_mesh(domain_box, style="wireframe", color="dimgray", line_width=1.5)
    
    # Pre-allocate cylinder actor.
    # compute_normals with auto_orient_normals=True corrects any inward-facing normals.
    vtk_faces = np.c_[np.full(len(MARKER_FACES), 3), MARKER_FACES].ravel()
    pv_mesh = pv.PolyData(MARKER_INITIAL_NP.copy(), vtk_faces)
    pv_mesh.compute_normals(cell_normals=False, point_normals=True,
                            consistent_normals=True, auto_orient_normals=True,
                            inplace=True)
    cyl_actor = pl.add_mesh(pv_mesh, color="white", smooth_shading=True,
                            specular=0.6, specular_power=30, ambient=0.1,
                            opacity=0.95)

    # Pre-allocate Q-criterion actor colored by ω_z (spanwise vorticity).
    # Seed with a tiny sphere carrying the 'omega_z' scalar so the
    # colormap / clim are committed to the actor before the first update.
    _seed = pv.Sphere(radius=0.001)
    _seed.point_data["omega_z"] = np.zeros(_seed.n_points)
    q_actor = pl.add_mesh(
        _seed,
        scalars="omega_z",
        cmap="RdBu_r",
        clim=(-OMEGA_Z_CLIM, OMEGA_Z_CLIM),
        smooth_shading=True,
        opacity=1,
        specular=0,
        show_scalar_bar=True,
    )

    pl.show(interactive_update=True, auto_close=False, window_size=[1280, 720])

step_idx = jnp.array(0, dtype=jnp.int32)
shedding_period = D / (0.2 * U0)  # approximate, for reporting frequency

with tqdm(total=TM, unit="step") as pbar:
    for n_steps in chunk_sizes:
        carry, traj = update_chunk((f, d, v, a, step_idx), n_steps)
        f, d, v, a, step_idx = carry
        jax.block_until_ready(f)

        if jnp.any(jnp.isnan(f)).item():
            tqdm.write(f"[ERROR] NaN detected at step {int(step_idx)} — simulation diverged!")
            break

        # traj shape: (n_steps, 2) = [[dy, hy], ...] — lift diagnostics
        dy_arr = np.asarray(traj[:, 0])
        hy_arr = np.asarray(traj[:, 1])
        hy_rms = float(np.sqrt(np.mean(hy_arr**2)))
        dy_max = float(np.max(np.abs(dy_arr)))

        # Show live diagnostics in the tqdm bar (no extra lines)
        pbar.set_postfix(
            Fy=f"{hy_rms:.3f}",
            A_D=f"{dy_max/D:.3f}",
            refresh=False,
        )
        pbar.update(n_steps)

        if PLOT:
            _, u_now = lbm3d.get_macroscopic(f)

            # Update Q-criterion isosurface colored by ω_z
            isosurf = get_q_criterion_isosurface(u_now, q_threshold=Q_THRESHOLD)
            q_actor.mapper.dataset.copy_from(isosurf)
            # copy_from may reset the active-scalar index; force omega_z again.
            q_actor.mapper.dataset.set_active_scalars("omega_z")

            # Update cylinder points in-place.
            # cyl_actor.mapper.dataset IS pv_mesh (same VTK object) — just
            # modify pv_mesh.points directly; no copy_from() needed and calling
            # it would trigger a "DeepCopy to itself" VTK warning.
            curr_d = np.asarray(d)
            pv_mesh.points[:] = MARKER_INITIAL_NP
            pv_mesh.points[:, 0] += curr_d[0]
            pv_mesh.points[:, 1] += curr_d[1]

            pl.update(stime=1)


if PLOT:
    pl.show()
