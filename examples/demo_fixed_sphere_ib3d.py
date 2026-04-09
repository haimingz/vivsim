"""Minimal 3D fixed-sphere coupling demo for ``ib3d`` + ``lbm3d``.

The setup is intentionally simple:
- 3D periodic box
- D3Q19 BGK fluid solver
- stationary spherical immersed boundary
- initially uniform flow in the positive x direction

The goal is not a production-quality benchmark. This script is meant as a
small end-to-end example that exercises the 3D LBM and IB modules together,
prints a few diagnostics, and exports an animated visualization.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from vivsim import ib3d, lbm3d


OUTPUT_DIR = Path("outputs")
DEFAULT_GIF = OUTPUT_DIR / "demo_fixed_sphere_ib3d_3d.gif"
DEFAULT_FIGURE = OUTPUT_DIR / "demo_fixed_sphere_ib3d_summary.png"
DEFAULT_GRID_N = 56
DEFAULT_RADIUS = 8.0
DEFAULT_STEPS = 84
DEFAULT_FRAME_EVERY = 2
DEFAULT_FPS = 10


@dataclass
class StepDiagnostics:
    step: int
    mean_rho: float
    max_speed: float
    marker_slip_l2: float
    drag_x: float
    lift_y: float
    lift_z: float


def compute_vorticity(u):
    """Return the 3D vorticity vector field."""

    ux, uy, uz = u

    dux_dy = jnp.gradient(ux, axis=1)
    dux_dz = jnp.gradient(ux, axis=2)
    duy_dx = jnp.gradient(uy, axis=0)
    duy_dz = jnp.gradient(uy, axis=2)
    duz_dx = jnp.gradient(uz, axis=0)
    duz_dy = jnp.gradient(uz, axis=1)

    wx = duz_dy - duy_dz
    wy = dux_dz - duz_dx
    wz = duy_dx - dux_dy
    return jnp.stack([wx, wy, wz], axis=0)


def resolve_slice_index(axis: str, slice_index: int | None, grid_shape):
    """Clamp the requested slice index to the selected grid axis."""

    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_id = axis_map[axis]
    axis_size = int(grid_shape[axis_id])

    if slice_index is None:
        return axis_size // 2

    return max(0, min(axis_size - 1, int(slice_index)))


def get_slice_field(field, axis: str, index: int):
    """Extract a 2D cross-section from a 3D scalar field."""

    if axis == "x":
        return field[index, :, :], ("y", "z")
    if axis == "y":
        return field[:, index, :], ("x", "z")
    if axis == "z":
        return field[:, :, index], ("x", "y")
    raise ValueError(f"Unsupported slice axis: {axis}")


def get_cross_section_circle(center, radius: float, axis: str, index: int):
    """Return the circle visible when a slice intersects the sphere."""

    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_id = axis_map[axis]
    distance = float(index) - float(center[axis_id])

    if abs(distance) >= radius:
        return None

    circle_radius = math.sqrt(radius**2 - distance**2)
    if axis == "x":
        return float(center[1]), float(center[2]), circle_radius
    if axis == "y":
        return float(center[0]), float(center[2]), circle_radius
    return float(center[0]), float(center[1]), circle_radius


def _add_circle_overlay(ax, center, radius: float, axis: str, index: int):
    """Draw the sphere cross-section on a 2D slice when it exists."""

    from matplotlib.patches import Circle

    circle = get_cross_section_circle(center, radius, axis, index)
    if circle is None:
        return None

    cx, cy, circle_radius = circle
    patch = Circle(
        (cx, cy),
        circle_radius,
        fill=True,
        facecolor="#dbeafe",
        linewidth=1.8,
        linestyle="-",
        edgecolor="#f8fafc",
        alpha=0.92,
        zorder=4,
    )
    ax.add_patch(patch)
    outline = Circle(
        (cx, cy),
        circle_radius,
        fill=False,
        linewidth=1.8,
        linestyle="--",
        edgecolor="#0f172a",
        alpha=0.95,
        zorder=5,
    )
    ax.add_patch(outline)
    return patch


def _compute_frame_data(step: int, u):
    """Collect visualization data for a single time step."""

    vorticity = compute_vorticity(u)
    vorticity_mag = jnp.linalg.norm(vorticity, axis=0)
    return {
        "step": int(step),
        "u": np.asarray(u, dtype=np.float32),
        "vorticity_mag": np.asarray(vorticity_mag, dtype=np.float32),
        "max_speed": float(jnp.max(jnp.linalg.norm(u, axis=0))),
    }


def get_slice_velocity_components(u, axis: str, index: int):
    """Return in-plane velocity components for a 2D slice."""

    if axis == "x":
        return np.asarray(u[1, index, :, :]), np.asarray(u[2, index, :, :]), ("y", "z")
    if axis == "y":
        return np.asarray(u[0, :, index, :]), np.asarray(u[2, :, index, :]), ("x", "z")
    if axis == "z":
        return np.asarray(u[0, :, :, index]), np.asarray(u[1, :, :, index]), ("x", "y")
    raise ValueError(f"Unsupported slice axis: {axis}")


def make_sphere_surface(center, radius: float, n_theta: int = 72, n_phi: int = 40):
    """Build a smooth sphere surface mesh for 3D plotting."""

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    phi = np.linspace(0.0, np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="xy")

    x = center[0] + radius * np.cos(theta_grid) * np.sin(phi_grid)
    y = center[1] + radius * np.sin(theta_grid) * np.sin(phi_grid)
    z = center[2] + radius * np.cos(phi_grid)
    return x, y, z


def _build_slice_animation(frames, center, radius: float, fps: int = 10, axis: str = "z", slice_index: int | None = None):
    import matplotlib.pyplot as plt
    from matplotlib import animation

    if not frames:
        raise ValueError("No frames were recorded for animation.")

    grid_shape = frames[0]["vorticity_mag"].shape
    slice_index = resolve_slice_index(axis, slice_index, grid_shape)
    speed_vmax = max(float(frame["max_speed"]) for frame in frames)
    speed_vmax = speed_vmax if speed_vmax > 0 else 1.0
    vort_vmax = max(float(frame["vorticity_mag"].max()) for frame in frames)
    vort_vmax = vort_vmax if vort_vmax > 0 else 1.0

    frame0 = frames[0]
    speed0 = np.linalg.norm(frame0["u"], axis=0)
    speed_slice0, labels = get_slice_field(speed0, axis, slice_index)
    vort_slice0, _ = get_slice_field(frame0["vorticity_mag"], axis, slice_index)
    vel_a0, vel_b0, _ = get_slice_velocity_components(frame0["u"], axis, slice_index)
    stride = max(1, min(speed_slice0.shape) // 14)
    arrow_scale = 0.08 * min(speed_slice0.shape) / max(speed_vmax, 1e-6)
    x_coords, y_coords = np.meshgrid(
        np.arange(speed_slice0.shape[0]),
        np.arange(speed_slice0.shape[1]),
        indexing="ij",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), constrained_layout=True)
    speed_image = axes[0].imshow(
        np.asarray(speed_slice0).T,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=speed_vmax,
        interpolation="bicubic",
    )
    vort_image = axes[1].imshow(
        np.asarray(vort_slice0).T,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=vort_vmax,
        interpolation="bicubic",
    )
    for ax in axes:
        _add_circle_overlay(ax, center, radius, axis, slice_index)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    axes[0].set_title("Speed With Slice Arrows")
    axes[1].set_title("Vorticity Magnitude")
    fig.colorbar(speed_image, ax=axes[0], label="|u|", shrink=0.86)
    fig.colorbar(vort_image, ax=axes[1], label="|omega|", shrink=0.86)

    quiver = axes[0].quiver(
        x_coords[::stride, ::stride],
        y_coords[::stride, ::stride],
        vel_a0[::stride, ::stride] * arrow_scale,
        vel_b0[::stride, ::stride] * arrow_scale,
        color="#f8fafc",
        alpha=0.92,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0032,
        zorder=6,
    )
    text = axes[0].text(
        0.02,
        0.98,
        "",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=10,
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 4},
    )

    def update(frame):
        speed = np.linalg.norm(frame["u"], axis=0)
        speed_slice, _ = get_slice_field(speed, axis, slice_index)
        vort_slice, _ = get_slice_field(frame["vorticity_mag"], axis, slice_index)
        vel_a, vel_b, _ = get_slice_velocity_components(frame["u"], axis, slice_index)

        speed_image.set_data(np.asarray(speed_slice).T)
        vort_image.set_data(np.asarray(vort_slice).T)
        quiver.set_UVC(
            vel_a[::stride, ::stride] * arrow_scale,
            vel_b[::stride, ::stride] * arrow_scale,
        )
        fig.suptitle(
            f"Fixed Sphere IB3D | {axis.upper()}={slice_index} | step={frame['step']}"
        )
        text.set_text(
            f"max |u| = {float(frame['max_speed']):.3e}\n"
            f"max |omega| = {float(frame['vorticity_mag'].max()):.3e}"
        )
        return speed_image, vort_image, quiver, text

    update(frames[0])
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / max(1, fps),
        blit=False,
    )
    return fig, ani


def _build_3d_animation(frames, center, radius: float, fps: int = 10, view_elev: float = 22.0, view_azim: float = 35.0, rotation_span: float = 1.0):
    import matplotlib.pyplot as plt
    from matplotlib import animation, cm, colors

    if not frames:
        raise ValueError("No frames were recorded for animation.")

    grid_shape = frames[0]["vorticity_mag"].shape
    nx, ny, nz = grid_shape
    x_mid = nx // 2
    y_mid = ny // 2
    z_mid = nz // 2
    speed_vmax = max(float(frame["max_speed"]) for frame in frames)
    speed_vmax = speed_vmax if speed_vmax > 0 else 1.0
    norm = colors.Normalize(vmin=0.0, vmax=speed_vmax)
    cmap = plt.colormaps["viridis"]

    x_xy, y_xy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    x_xz, z_xz = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    y_yz, z_yz = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
    sphere_x, sphere_y, sphere_z = make_sphere_surface(
        center,
        radius,
        n_theta=max(72, int(round(8 * radius))),
        n_phi=max(40, int(round(4 * radius))),
    )
    quiver_stride = max(2, min(nx, nz) // 12)
    arrow_scale = 0.08 * min(nx, ny, nz) / max(speed_vmax, 1e-6)
    xq = x_xz[::quiver_stride, ::quiver_stride]
    zq = z_xz[::quiver_stride, ::quiver_stride]
    yq = np.full_like(xq, y_mid, dtype=float)

    fig = plt.figure(figsize=(8.2, 6.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    colorbar = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(colorbar, ax=ax, shrink=0.68, pad=0.08, label="|u|")

    def update(frame_index):
        frame = frames[frame_index]
        u = np.asarray(frame["u"])
        speed = np.linalg.norm(u, axis=0)
        angle = view_azim + rotation_span * 120.0 * frame_index / max(1, len(frames) - 1)

        ax.clear()
        ax.plot_surface(
            x_xy,
            y_xy,
            np.full_like(x_xy, z_mid, dtype=float),
            facecolors=cmap(norm(speed[:, :, z_mid])),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=True,
            alpha=0.74,
        )
        ax.plot_surface(
            x_xz,
            np.full_like(x_xz, y_mid, dtype=float),
            z_xz,
            facecolors=cmap(norm(speed[:, y_mid, :])),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=True,
            alpha=0.42,
        )
        ax.plot_surface(
            np.full_like(y_yz, x_mid, dtype=float),
            y_yz,
            z_yz,
            facecolors=cmap(norm(speed[x_mid, :, :])),
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            shade=True,
            alpha=0.32,
        )
        ax.plot_surface(
            sphere_x,
            sphere_y,
            sphere_z,
            color="#dbeafe",
            edgecolor="#475569",
            linewidth=0.18,
            antialiased=True,
            alpha=0.97,
            shade=True,
            zorder=6,
        )
        ax.quiver(
            xq,
            yq,
            zq,
            u[0, ::quiver_stride, y_mid, ::quiver_stride] * arrow_scale,
            np.zeros_like(xq),
            u[2, ::quiver_stride, y_mid, ::quiver_stride] * arrow_scale,
            color="#f8fafc",
            alpha=0.80,
            linewidth=0.7,
        )
        ax.text2D(
            0.02,
            0.96,
            (
                f"step={frame['step']}  "
                f"max |u|={float(frame['max_speed']):.3e}  "
                f"max |omega|={float(frame['vorticity_mag'].max()):.3e}"
            ),
            transform=ax.transAxes,
            color="#0f172a",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.82, "pad": 4},
        )

        ax.set_xlim(0, nx - 1)
        ax.set_ylim(0, ny - 1)
        ax.set_zlim(0, nz - 1)
        ax.set_box_aspect((nx, ny, nz))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=view_elev, azim=angle)
        ax.set_title("Fixed Sphere IB3D | Speed Slices + Mid-Plane Flow Arrows")

        return ()

    update(0)
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / max(1, fps),
        blit=False,
    )
    return fig, ani


def save_animation(frames, output_path, center, radius: float, fps: int = 10, view: str = "3d", axis: str = "z", slice_index: int | None = None, grid_shape=None):
    """Save a GIF animation for either the 3D or slice view."""

    from matplotlib import animation
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if view == "2d":
        fig, ani = _build_slice_animation(
            frames,
            center=center,
            radius=radius,
            fps=fps,
            axis=axis,
            slice_index=slice_index,
        )
    else:
        fig, ani = _build_3d_animation(
            frames,
            center=center,
            radius=radius,
            fps=fps,
        )

    ani.save(str(output_path), writer=animation.PillowWriter(fps=max(1, fps)))
    plt.close(fig)
    return output_path


def show_animation_window(frames, center, radius: float, fps: int = 10, view: str = "3d", axis: str = "z", slice_index: int | None = None):
    """Show the animation in a matplotlib window."""

    import matplotlib.pyplot as plt

    if view == "2d":
        _build_slice_animation(
            frames,
            center=center,
            radius=radius,
            fps=fps,
            axis=axis,
            slice_index=slice_index,
        )
    else:
        _build_3d_animation(frames, center=center, radius=radius, fps=fps)

    plt.show()


def open_output_path(path):
    """Open a generated file with the platform default handler."""

    if path is None:
        return

    path = Path(path)
    if not path.exists():
        return

    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def _plot_scalar_panel(ax, field, title: str, center, radius: float, axis: str, index: int, cmap: str, with_colorbar: bool = False):
    image_field, labels = get_slice_field(field, axis, index)
    image = ax.imshow(
        np.asarray(image_field).T,
        origin="lower",
        cmap=cmap,
        interpolation="bicubic",
    )
    _add_circle_overlay(ax, center, radius, axis, index)
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    return image if with_colorbar else None


def _plot_speed_panel(ax, u, title: str, center, radius: float, axis: str, index: int):
    speed = np.linalg.norm(np.asarray(u), axis=0)
    image_field, labels = get_slice_field(speed, axis, index)
    vel_a, vel_b, _ = get_slice_velocity_components(u, axis, index)

    image = ax.imshow(
        np.asarray(image_field).T,
        origin="lower",
        cmap="viridis",
        interpolation="bicubic",
    )
    _add_circle_overlay(ax, center, radius, axis, index)

    stride = max(1, min(image_field.shape) // 14)
    arrow_scale = 0.08 * min(image_field.shape) / max(float(np.max(speed)), 1e-6)
    x_coords, y_coords = np.meshgrid(
        np.arange(image_field.shape[0]),
        np.arange(image_field.shape[1]),
        indexing="ij",
    )
    ax.quiver(
        x_coords[::stride, ::stride],
        y_coords[::stride, ::stride],
        vel_a[::stride, ::stride] * arrow_scale,
        vel_b[::stride, ::stride] * arrow_scale,
        color="#f8fafc",
        alpha=0.90,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0032,
        zorder=6,
    )
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    return image


def save_summary_figure(rho, u, center, radius: float, output_path, step: int, u0: float, diagnostics: StepDiagnostics, xy_index: int, xz_index: int):
    """Save a compact summary figure with final slices and diagnostics."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vorticity_mag = np.asarray(jnp.linalg.norm(compute_vorticity(u), axis=0))
    rho = np.asarray(rho)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    speed_xy = _plot_speed_panel(
        axes[0, 0],
        u,
        f"Speed + Arrows | XY @ z={xy_index}",
        center=center,
        radius=radius,
        axis="z",
        index=xy_index,
    )
    speed_xz = _plot_speed_panel(
        axes[0, 1],
        u,
        f"Speed + Arrows | XZ @ y={xz_index}",
        center=center,
        radius=radius,
        axis="y",
        index=xz_index,
    )
    vort_xy = _plot_scalar_panel(
        axes[1, 0],
        vorticity_mag,
        f"|omega| | XY @ z={xy_index}",
        center=center,
        radius=radius,
        axis="z",
        index=xy_index,
        cmap="magma",
        with_colorbar=True,
    )
    vort_xz = _plot_scalar_panel(
        axes[1, 1],
        vorticity_mag,
        f"|omega| | XZ @ y={xz_index}",
        center=center,
        radius=radius,
        axis="y",
        index=xz_index,
        cmap="magma",
        with_colorbar=True,
    )

    fig.colorbar(speed_xy, ax=axes[0, :], shrink=0.78, label="|u|")
    fig.colorbar(vort_xy, ax=axes[1, :], shrink=0.78, label="|omega|")

    fig.suptitle(
        "Fixed Sphere IB3D Summary\n"
        f"step={step} | rho_mean={rho.mean():.6f} | "
        f"u0={u0:.4f} | u_max={diagnostics.max_speed:.4f} | "
        f"slip_l2={diagnostics.marker_slip_l2:.3e}\n"
        f"drag=({diagnostics.drag_x:.3e}, {diagnostics.lift_y:.3e}, {diagnostics.lift_z:.3e})"
    )

    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def make_fibonacci_sphere(center, radius: float, n_markers: int):
    """Generate quasi-uniform markers on the sphere surface."""

    n_markers = int(n_markers)
    if n_markers <= 0:
        raise ValueError("n_markers must be positive.")

    center = jnp.asarray(center, dtype=jnp.float32)
    indices = jnp.arange(n_markers, dtype=jnp.float32) + 0.5
    phi = jnp.arccos(1.0 - 2.0 * indices / n_markers)
    theta = jnp.pi * (3.0 - jnp.sqrt(5.0)) * indices

    sphere = jnp.stack(
        [
            jnp.cos(theta) * jnp.sin(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(phi),
        ],
        axis=1,
    )
    return center[None, :] + radius * sphere


def compute_marker_area_weights(radius: float, n_markers: int):
    """Approximate uniform area weights for the surface markers."""

    total_area = 4.0 * math.pi * radius**2
    return jnp.full((int(n_markers),), total_area / int(n_markers), dtype=jnp.float32)


def compute_diagnostics(step: int, rho, u, marker_u, marker_u_target, marker_reaction_force):
    """Compute a small set of scalar diagnostics for logging."""

    speed = jnp.linalg.norm(u, axis=0)
    marker_slip = marker_u - marker_u_target
    marker_slip_l2 = jnp.sqrt(jnp.mean(jnp.sum(marker_slip**2, axis=1)))
    total_force = jnp.sum(marker_reaction_force, axis=0)

    return StepDiagnostics(
        step=int(step),
        mean_rho=float(jnp.mean(rho)),
        max_speed=float(jnp.max(speed)),
        marker_slip_l2=float(marker_slip_l2),
        drag_x=float(total_force[0]),
        lift_y=float(total_force[1]),
        lift_z=float(total_force[2]),
    )


def run_fixed_sphere_test(nx: int = DEFAULT_GRID_N, ny: int = DEFAULT_GRID_N, nz: int = DEFAULT_GRID_N, radius: float = DEFAULT_RADIUS, n_markers: int | None = None, steps: int = DEFAULT_STEPS, nu: float = 0.1, u0: float = 0.03, drive_x: float = 0.0, mdf_iter: int = 5, report_every: int = 5, record_frames: bool = False, frame_every: int = DEFAULT_FRAME_EVERY, slice_axis: str = "z", slice_index: int | None = None, animation_path=None, figure_path=None, show_animation: bool = False, open_output: bool = False, animation_view: str = "3d", fps: int = DEFAULT_FPS):
    """Run a minimal fixed-sphere IB/LBM coupling test and print diagnostics."""

    if steps <= 0:
        raise ValueError("steps must be positive.")

    if min(nx, ny, nz) < 12:
        raise ValueError("Grid size is too small for the fixed-sphere demo.")

    slice_axis = slice_axis.lower()
    if slice_axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported slice axis: {slice_axis}")

    if animation_view not in {"2d", "3d"}:
        raise ValueError("animation_view must be either '2d' or '3d'.")

    grid_shape = (int(nx), int(ny), int(nz))
    center = jnp.array(
        [(grid_shape[0] - 1) * 0.5, (grid_shape[1] - 1) * 0.5, (grid_shape[2] - 1) * 0.5],
        dtype=jnp.float32,
    )

    max_radius = 0.5 * (min(grid_shape) - 6)
    if radius <= 0 or radius >= max_radius:
        raise ValueError(
            f"radius must be in (0, {max_radius:.2f}) for grid {grid_shape}, got {radius}."
        )

    if n_markers is None:
        n_markers = max(480, int(round(8.0 * math.pi * radius**2)))

    report_every = max(1, int(report_every))
    frame_every = max(1, int(frame_every))
    slice_index = resolve_slice_index(slice_axis, slice_index, grid_shape)

    rho = jnp.ones(grid_shape, dtype=jnp.float32)
    u = jnp.zeros((3, *grid_shape), dtype=jnp.float32)
    u = u.at[0].set(u0)
    f = lbm3d.get_equilibrium(rho, u)
    omega = lbm3d.get_omega(nu)

    marker_coords = make_fibonacci_sphere(center, radius, n_markers)
    marker_dA = compute_marker_area_weights(radius, n_markers)
    stencil_weights, stencil_indices = ib3d.get_ib_stencil(marker_coords, grid_shape)
    marker_u_target = jnp.zeros((n_markers, 3), dtype=jnp.float32)

    drive_force = jnp.zeros_like(u)
    if drive_x != 0.0:
        drive_force = drive_force.at[0].set(drive_x)

    store_frames = record_frames or animation_path is not None or show_animation
    frames = []
    last_diag = None

    for step in range(steps):
        rho, u = lbm3d.get_macroscopic(f)

        grid_force_ib, marker_reaction_force = ib3d.multi_direct_forcing(
            grid_u=u,
            stencil_weights=stencil_weights,
            stencil_indices=stencil_indices,
            marker_u_target=marker_u_target,
            marker_dA=marker_dA,
            n_iter=mdf_iter,
        )
        total_force = grid_force_ib + drive_force

        feq = lbm3d.get_equilibrium(rho, u)
        f = lbm3d.collision_bgk(f, feq, omega)
        f = lbm3d.forcing_guo_bgk(f, total_force, u, omega)
        f = lbm3d.streaming(f)

        rho, u = lbm3d.get_macroscopic(f)
        marker_u = ib3d.interpolate(u, stencil_weights, stencil_indices)
        last_diag = compute_diagnostics(
            step=step,
            rho=rho,
            u=u,
            marker_u=marker_u,
            marker_u_target=marker_u_target,
            marker_reaction_force=marker_reaction_force,
        )

        if step % report_every == 0 or step == steps - 1:
            print(
                f"step={step:04d} "
                f"rho_mean={last_diag.mean_rho:.6f} "
                f"u_max={last_diag.max_speed:.6f} "
                f"slip_l2={last_diag.marker_slip_l2:.6e} "
                f"drag=({last_diag.drag_x:.6e}, {last_diag.lift_y:.6e}, {last_diag.lift_z:.6e})"
            )

        if store_frames and (step % frame_every == 0 or step == steps - 1):
            frames.append(_compute_frame_data(step, u))

    center_np = tuple(float(value) for value in np.asarray(center))
    saved_animation = None
    saved_figure = None

    if animation_path is not None and frames:
        saved_animation = save_animation(
            frames=frames,
            output_path=animation_path,
            center=center_np,
            radius=radius,
            fps=fps,
            view=animation_view,
            axis=slice_axis,
            slice_index=slice_index,
            grid_shape=grid_shape,
        )

    if figure_path is not None and last_diag is not None:
        saved_figure = save_summary_figure(
            rho=rho,
            u=u,
            center=center_np,
            radius=radius,
            output_path=figure_path,
            step=steps - 1,
            u0=u0,
            diagnostics=last_diag,
            xy_index=resolve_slice_index("z", None, grid_shape),
            xz_index=resolve_slice_index("y", None, grid_shape),
        )

    if show_animation and frames:
        show_animation_window(
            frames=frames,
            center=center_np,
            radius=radius,
            fps=fps,
            view=animation_view,
            axis=slice_axis,
            slice_index=slice_index,
        )

    if open_output:
        open_output_path(saved_animation)
        open_output_path(saved_figure)

    return {
        "grid_shape": grid_shape,
        "radius": radius,
        "n_markers": n_markers,
        "steps": steps,
        "omega": float(omega),
        "last_diagnostics": last_diag,
        "slice_axis": slice_axis,
        "slice_index": int(slice_index),
        "animation_path": None if saved_animation is None else str(saved_animation),
        "figure_path": None if saved_figure is None else str(saved_figure),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a fixed-sphere ib3d + lbm3d demo."
    )
    parser.add_argument("--nx", type=int, default=DEFAULT_GRID_N)
    parser.add_argument("--ny", type=int, default=DEFAULT_GRID_N)
    parser.add_argument("--nz", type=int, default=DEFAULT_GRID_N)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--n-markers", type=int, default=None)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--u0", type=float, default=0.03)
    parser.add_argument("--drive-x", type=float, default=0.0)
    parser.add_argument("--mdf-iter", type=int, default=5)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--save-gif", type=str, default=str(DEFAULT_GIF))
    parser.add_argument("--no-save-gif", dest="save_gif", action="store_const", const=None)
    parser.add_argument("--save-figure", type=str, default=str(DEFAULT_FIGURE))
    parser.add_argument("--no-save-figure", dest="save_figure", action="store_const", const=None)
    parser.add_argument("--show-animation", action="store_true", dest="show_animation")
    parser.add_argument("--open-output", action="store_true")
    parser.add_argument("--frame-every", type=int, default=DEFAULT_FRAME_EVERY)
    parser.add_argument("--animation-view", choices=("3d", "2d"), default="3d")
    parser.add_argument("--slice-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_fixed_sphere_test(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        radius=args.radius,
        n_markers=args.n_markers,
        steps=args.steps,
        nu=args.nu,
        u0=args.u0,
        drive_x=args.drive_x,
        mdf_iter=args.mdf_iter,
        report_every=args.report_every,
        record_frames=args.save_gif is not None or args.show_animation,
        frame_every=args.frame_every,
        slice_axis=args.slice_axis,
        slice_index=args.slice_index,
        animation_path=args.save_gif,
        figure_path=args.save_figure,
        show_animation=args.show_animation,
        open_output=args.open_output,
        animation_view=args.animation_view,
        fps=args.fps,
    )

    diag = result["last_diagnostics"]
    print(
        "\nfinal: "
        f"grid={result['grid_shape']} "
        f"radius={result['radius']:.2f} "
        f"markers={result['n_markers']} "
        f"steps={result['steps']} "
        f"omega={result['omega']:.6f} "
        f"slip_l2={diag.marker_slip_l2:.6e}"
    )
    if result["animation_path"] is not None:
        print(f"saved animation: {result['animation_path']}")
    if result["figure_path"] is not None:
        print(f"saved summary: {result['figure_path']}")


if __name__ == "__main__":
    main()
