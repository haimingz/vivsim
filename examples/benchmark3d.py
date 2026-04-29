"""
Performance benchmark for vivsim 3D LBM functions.

This script measures the execution time of core D3Q19 LBM and 3D IB functions
and reports the time consumption ratio, making it easy to identify performance
bottlenecks.

Usage:
    python examples/benchmark_3d.py
    python examples/benchmark_3d.py --nx 192 --ny 128 --nz 128 --markers 642

Requirements:
    pip install vivsim
"""

import argparse
import math
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from vivsim import ib3d, lbm3d


# ====================== Geometry helper ======================

def icosphere(radius, center, subdivisions=2):
    """Generate a simple icosphere triangulation for IB benchmarks."""

    phi = (1 + math.sqrt(5)) / 2
    verts = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float32,
    )
    verts /= np.linalg.norm(verts[0])

    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    for _ in range(subdivisions):
        edge_midpoint = {}
        new_faces = []
        for tri in faces:
            mids = []
            for i in range(3):
                edge = tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
                if edge not in edge_midpoint:
                    mid = (verts[edge[0]] + verts[edge[1]]) * 0.5
                    mid /= np.linalg.norm(mid)
                    edge_midpoint[edge] = len(verts)
                    verts = np.vstack([verts, mid])
                mids.append(edge_midpoint[edge])
            a, b, c = tri
            m0, m1, m2 = mids
            new_faces.extend(
                [
                    [a, m0, m2],
                    [b, m1, m0],
                    [c, m2, m1],
                    [m0, m1, m2],
                ]
            )
        faces = np.asarray(new_faces, dtype=np.int32)

    verts = verts * radius + np.asarray(center, dtype=np.float32)
    return verts, faces


def pick_subdivisions(target_markers):
    """Return the icosphere subdivision level closest to target_markers."""

    candidates = [(level, 10 * 4**level + 2) for level in range(6)]
    return min(candidates, key=lambda item: abs(item[1] - target_markers))[0]


# ====================== Timing helper ======================

def measure(fn, args, repeat):
    jitted_fn = jax.jit(fn)
    warmup_result = jitted_fn(*args)
    jax.block_until_ready(warmup_result)

    can_feed_back = (
        isinstance(warmup_result, jax.Array)
        and isinstance(args[0], jax.Array)
        and warmup_result.shape == args[0].shape
    )

    if not can_feed_back:
        start = time.perf_counter()
        for _ in range(repeat):
            result = jitted_fn(*args)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start

        return elapsed / repeat * 1e6

    @jax.jit
    def benchmark_loop(args_tuple):
        def body(carry, _):
            result = fn(*carry)

            # Feed same-shape array results back into the loop so repeated calls
            # cannot be optimized away.
            return (result,) + carry[1:], None

        final_carry, _ = jax.lax.scan(body, args_tuple, None, length=repeat)
        return final_carry

    warmup_result = benchmark_loop(args)
    jax.block_until_ready(warmup_result)

    start = time.perf_counter()
    result = benchmark_loop(args)
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - start

    return elapsed / repeat * 1e6


# ====================== Benchmark ======================

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark vivsim 3D operators.")
    parser.add_argument("--nx", type=int, default=128, help="grid size in x direction")
    parser.add_argument("--ny", type=int, default=128, help="grid size in y direction")
    parser.add_argument("--nz", type=int, default=128, help="grid size in z direction")
    parser.add_argument(
        "--markers",
        type=int,
        default=2562,
        help="target number of IB markers; nearest icosphere size is used",
    )
    parser.add_argument("--repeat", type=int, default=50, help="timed iterations")
    parser.add_argument("--radius", type=float, default=24.0, help="IB sphere radius")
    return parser.parse_args()


def main():
    args = parse_args()
    nx, ny, nz = args.nx, args.ny, args.nz
    repeat = args.repeat

    # ====================== Initialize with meaningless values ======================

    nu = 0.01
    omega = lbm3d.get_omega(nu)

    rho = jnp.ones((nx, ny, nz), dtype=jnp.float32)
    u = jnp.zeros((3, nx, ny, nz), dtype=jnp.float32)
    f = lbm3d.get_equilibrium(rho, u)
    feq = lbm3d.get_equilibrium(rho, u)
    g = jnp.full((3, nx, ny, nz), 1e-5, dtype=jnp.float32)

    mask = jnp.zeros((nx, ny, nz), dtype=bool)
    mrt_op = lbm3d.get_mrt_collision_operator(omega)
    mrt_fop = lbm3d.get_mrt_forcing_operator(omega)

    subdivisions = pick_subdivisions(args.markers)
    radius = min(args.radius, 0.2 * min(nx, ny, nz))
    marker_coords_np, marker_faces_np = icosphere(
        radius=radius,
        center=(0.5 * nx, 0.5 * ny, 0.5 * nz),
        subdivisions=subdivisions,
    )
    marker_coords = jnp.asarray(marker_coords_np, dtype=jnp.float32)
    marker_faces = jnp.asarray(marker_faces_np, dtype=jnp.int32)
    marker_dA = ib3d.get_ds(marker_coords, marker_faces)
    marker_u_zero = jnp.zeros((marker_coords.shape[0], 3), dtype=jnp.float32)
    r_vals = jnp.linspace(0, 2.5, marker_coords.shape[0], dtype=jnp.float32)
    ib_stencil_weights, ib_stencil_indices = ib3d.get_ib_stencil(
        marker_coords,
        grid_shape=(nx, ny, nz),
        kernel=ib3d.kernel_peskin_4pt,
    )

    print(
        f"\nBenchmarking on {jax.devices()[0]} | grid: {nx}x{ny}x{nz} | "
        f"markers: {marker_coords.shape[0]} | repeats: {repeat}\n"
    )

    timings = {}

    # ------ Basic ------
    timings["lbm3d.get_macroscopic"] = measure(lbm3d.get_macroscopic, (f,), repeat)
    timings["lbm3d.get_equilibrium"] = measure(lbm3d.get_equilibrium, (rho, u), repeat)
    timings["lbm3d.streaming"] = measure(lbm3d.streaming, (f,), repeat)

    # ------ Collision ------
    timings["lbm3d.collision_bgk"] = measure(lbm3d.collision_bgk, (f, feq, omega), repeat)
    timings["lbm3d.collision_kbc"] = measure(lbm3d.collision_kbc, (f, feq, omega), repeat)
    timings["lbm3d.collision_mrt"] = measure(lbm3d.collision_mrt, (f, feq, mrt_op), repeat)
    timings["lbm3d.collision_reg"] = measure(lbm3d.collision_reg, (f, feq, omega), repeat)

    # ------ Forcing ------
    timings["lbm3d.forcing_edm"] = measure(lbm3d.forcing_edm, (f, g, u), repeat)
    timings["lbm3d.forcing_guo_bgk"] = measure(
        lbm3d.forcing_guo_bgk, (f, g, u, omega), repeat
    )
    timings["lbm3d.forcing_guo_mrt"] = measure(
        lbm3d.forcing_guo_mrt, (f, g, u, mrt_fop), repeat
    )
    timings["lbm3d.get_guo_forcing_term"] = measure(
        lbm3d.get_guo_forcing_term, (g, u), repeat
    )

    # ------ Boundary: NEE ------
    timings["lbm3d.boundary_nee"] = measure(partial(lbm3d.boundary_nee, loc="left"), (f,), repeat)
    timings["lbm3d.boundary_velocity_nee"] = measure(
        partial(lbm3d.boundary_velocity_nee, loc="left"), (f,), repeat
    )
    timings["lbm3d.boundary_pressure_nee"] = measure(
        partial(lbm3d.boundary_pressure_nee, loc="left"), (f,), repeat
    )

    # ------ Boundary: NEBB ------
    timings["lbm3d.boundary_nebb"] = measure(partial(lbm3d.boundary_nebb, loc="left"), (f,), repeat)
    timings["lbm3d.boundary_velocity_nebb"] = measure(
        partial(lbm3d.boundary_velocity_nebb, loc="left"), (f,), repeat
    )
    timings["lbm3d.boundary_pressure_nebb"] = measure(
        partial(lbm3d.boundary_pressure_nebb, loc="left"), (f,), repeat
    )

    # ------ Boundary: Equilibrium ------
    timings["lbm3d.boundary_equilibrium"] = measure(
        partial(lbm3d.boundary_equilibrium, loc="left"), (f,), repeat
    )

    # ------ Boundary: Bounce-back & specular reflection ------
    timings["lbm3d.boundary_bounce_back"] = measure(
        partial(lbm3d.boundary_bounce_back, loc="left"), (f, f), repeat
    )
    timings["lbm3d.boundary_specular_reflection"] = measure(
        partial(lbm3d.boundary_specular_reflection, loc="left"), (f, f), repeat
    )
    timings["lbm3d.obstacle_bounce_back"] = measure(
        partial(lbm3d.obstacle_bounce_back, mask=mask), (f,), repeat
    )

    # ------ Boundary: Characteristic ------
    timings["lbm3d.boundary_characteristic"] = measure(
        partial(lbm3d.boundary_characteristic, loc="right"), (rho, u), repeat
    )

    # ------ IB3D: geometry ------
    timings["ib3d.get_triangle_areas"] = measure(
        ib3d.get_triangle_areas, (marker_coords, marker_faces), repeat
    )
    timings["ib3d.get_surface_area"] = measure(
        ib3d.get_surface_area, (marker_coords, marker_faces), repeat
    )
    timings["ib3d.get_volume"] = measure(
        ib3d.get_volume, (marker_coords, marker_faces), repeat
    )
    timings["ib3d.get_ds"] = measure(
        ib3d.get_ds, (marker_coords, marker_faces), repeat
    )

    # ------ IB3D: kernels ------
    timings["ib3d.kernel_peskin_3pt"] = measure(ib3d.kernel_peskin_3pt, (r_vals,), repeat)
    timings["ib3d.kernel_peskin_4pt"] = measure(ib3d.kernel_peskin_4pt, (r_vals,), repeat)
    timings["ib3d.kernel_cosine_4pt"] = measure(ib3d.kernel_cosine_4pt, (r_vals,), repeat)

    # ------ IB3D: stencil ------
    timings["ib3d.get_ib_stencil"] = measure(
        partial(
            ib3d.get_ib_stencil,
            grid_shape=(nx, ny, nz),
            kernel=ib3d.kernel_peskin_4pt,
            stencil_radius=2,
        ),
        (marker_coords,),
        repeat,
    )
    timings["ib3d.interpolate"] = measure(
        ib3d.interpolate, (u, ib_stencil_weights, ib_stencil_indices), repeat
    )
    timings["ib3d.spread"] = measure(
        ib3d.spread,
        (marker_u_zero, u, ib_stencil_weights, ib_stencil_indices),
        repeat,
    )

    # ------ IB3D: multi-direct forcing ------
    timings["ib3d.multi_direct_forcing"] = measure(
        ib3d.multi_direct_forcing,
        (u, ib_stencil_weights, ib_stencil_indices, marker_u_zero, marker_dA),
        repeat,
    )

    # ====================== Report results ======================

    col_w = 38
    bar_width = 20
    max_t = max(timings.values())
    scale = max_t / bar_width if max_t > 0 else 1.0

    print(f"{'Function':<{col_w}} {'Time (us)':>10}  Bar")
    print("-" * (col_w + 45))
    for name, t in timings.items():
        bar = "=" * int(t / scale)
        print(f"{name:<{col_w}} {t:>10.3f}  {bar}")
    print("-" * (col_w + 45))
    print(f"{'Functions count':<{col_w}} {len(timings):>10}")
    print()


if __name__ == "__main__":
    main()
