"""
Performance benchmark for vivsim LBM functions.

This script measures the execution time of each core LBM function and reports
the time consumption ratio, making it easy to identify performance bottlenecks.

Usage:
    python examples/benchmark.py

Requirements:
    pip install vivsim
"""

import time
from functools import partial

import jax
import jax.numpy as jnp

from vivsim import ib, lbm

# ====================== Configuration ======================

NX = 1024           # grid size in x direction
NY = 1024           # grid size in y direction
N_MARKER = 512      # number of IB markers (for IB-related benchmarks)
N_WARMUP = 10       # number of warm-up iterations (for JIT compilation)
N_REPEAT = 200      # number of timed iterations for each function

# ====================== Initialize with meaningless values ======================

u0 = 0.1
nu = 0.01
omega = lbm.get_omega(nu)

rho = jnp.ones((NX, NY))
u = jnp.zeros((2, NX, NY))
f = lbm.get_equilibrium(rho, u)
feq = lbm.get_equilibrium(rho, u)
g = jnp.full((2, NX, NY), 1e-5)          # body force density

# Extra inputs reused across benchmarks
mask = jnp.zeros((NX, NY), dtype=bool)    # obstacle mask (empty)
mrt_op  = lbm.get_mrt_collision_operator(omega)
mrt_fop = lbm.get_mrt_forcing_operator(omega)

# IB fixtures: circular cylinder with N_MARKER markers

theta = jnp.linspace(0, 2 * jnp.pi, N_MARKER, endpoint=False)
marker_x = NX / 2 + 50 * jnp.cos(theta)
marker_y = NY / 2 + 50 * jnp.sin(theta)
marker_coords = jnp.stack([marker_x, marker_y], axis=1)
marker_ds = ib.get_ds(marker_coords)
marker_u_zero = jnp.zeros((N_MARKER, 2))
r_vals = jnp.linspace(0, 2.5, N_MARKER)
ib_stencil_weights, ib_stencil_indices = ib.get_ib_stencil(marker_x, marker_y, NY)

# ====================== Timing helper ======================

def measure(fn, *args):   
    @jax.jit
    def benchmark_loop(args_tuple):
        def body(carry, _):
            result = fn(*carry)

            # Force memory operations by feeding the result back into the loop
            # as the new input, provided it has the exact same shape.
            is_array = isinstance(result, jax.Array) and isinstance(carry[0], jax.Array)
            if is_array and result.shape == carry[0].shape:
                carry = (result,) + carry[1:]
                
            return carry, result
        
        final_carry, _ = jax.lax.scan(body, args_tuple, None, length=N_REPEAT)
        return final_carry

    # Warm-up (we only need a few warm-ups now, the whole loop JITs once)
    _ = benchmark_loop(args)
    jax.block_until_ready(_)

    start = time.perf_counter()
    _ = benchmark_loop(args)
    jax.block_until_ready(_)
    elapsed = time.perf_counter() - start
    
    return elapsed / N_REPEAT * 1e6  # us per call


# ====================== Benchmark functions ======================

print(f"\nBenchmarking on {jax.devices()[0]} | grid: {NX}x{NY} | markers: {N_MARKER} | repeats: {N_REPEAT}\n")

timings = {}

# ------ Basic ------
timings["lbm.get_macroscopic"]    = measure(lbm.get_macroscopic, f)
timings["lbm.get_equilibrium"]    = measure(lbm.get_equilibrium, rho, u)
timings["lbm.streaming"]          = measure(lbm.streaming, f)

# ------ Collision ------
timings["lbm.collision_bgk"]         = measure(lbm.collision_bgk, f, feq, omega)
timings["lbm.collision_kbc"]         = measure(lbm.collision_kbc, f, feq, omega)
timings["lbm.collision_mrt"]         = measure(lbm.collision_mrt, f, feq, mrt_op)
timings["lbm.collision_reg"] = measure(lbm.collision_reg, f, feq, omega)

# ------ Forcing ------
timings["lbm.forcing_edm"]           = measure(lbm.forcing_edm, f, g, u)
timings["lbm.forcing_guo_bgk"]       = measure(lbm.forcing_guo_bgk, f, g, u, omega)
timings["lbm.forcing_guo_mrt"]       = measure(lbm.forcing_guo_mrt, f, g, u, mrt_fop)
timings["lbm.get_guo_forcing_term"]  = measure(lbm.get_guo_forcing_term, g, u)

# ------ Boundary: NEE ------
timings["lbm.boundary_nee"]          = measure(partial(lbm.boundary_nee, loc="left"), f)
timings["lbm.boundary_velocity_nee"] = measure(partial(lbm.boundary_velocity_nee, loc="left"), f)
timings["lbm.boundary_pressure_nee"] = measure(partial(lbm.boundary_pressure_nee, loc="left"), f)

# ------ Boundary: NEBB ------
timings["lbm.boundary_nebb"]          = measure(partial(lbm.boundary_force_corrected_nebb, loc="left"), f)
timings["lbm.boundary_velocity_nebb"] = measure(partial(lbm.boundary_velocity_nebb, loc="left"), f)
timings["lbm.boundary_pressure_nebb"] = measure(partial(lbm.boundary_pressure_nebb, loc="left"), f)

# ------ Boundary: Equilibrium ------
timings["lbm.boundary_equilibrium"] = measure(partial(lbm.boundary_equilibrium, loc="left"), f)

# ------ Boundary: Bounce-back & specular reflection ------
timings["lbm.boundary_bounce_back"]       = measure(partial(lbm.boundary_bounce_back, loc="left"), f, f)
timings["lbm.boundary_specular_reflection"] = measure(partial(lbm.boundary_specular_reflection, loc="left"), f, f)
timings["lbm.obstacle_bounce_back"]       = measure(partial(lbm.obstacle_bounce_back, mask=mask), f)

# ------ Boundary: Characteristic ------
timings["lbm.boundary_characteristic"] = measure(partial(lbm.boundary_characteristic, loc="right"), rho, u)

# ------ IB: geometry ------
timings["ib.get_area"]      = measure(ib.get_area, marker_coords) 
timings["ib.get_ds (closed)"] = measure(ib.get_ds, marker_coords)
timings["ib.get_ds (open)"]   = measure(partial(ib.get_ds, closed=False), marker_coords)

# ------ IB: kernels (input: 1-D distance array) ------
timings["ib.kernel_peskin_3pt"] = measure(ib.kernel_peskin_3pt, r_vals)
timings["ib.kernel_peskin_4pt"] = measure(ib.kernel_peskin_4pt, r_vals)
timings["ib.kernel_cosine_4pt"] = measure(ib.kernel_cosine_4pt, r_vals)

# ------ IB: stencil ------
timings["ib.get_ib_stencil"] = measure(partial(ib.get_ib_stencil, ny=NY, kernel=ib.kernel_peskin_4pt, stencil_radius=2), marker_x, marker_y)
timings["ib.interpolate"]    = measure(ib.interpolate, u, ib_stencil_weights, ib_stencil_indices)
timings["ib.spread"]         = measure(ib.spread, marker_u_zero, u, ib_stencil_weights, ib_stencil_indices)

# ------ IB: multi-direct forcing ------
timings["ib.multi_direct_forcing"] = measure(ib.multi_direct_forcing, u, ib_stencil_weights, ib_stencil_indices, marker_u_zero, marker_ds)

# ====================== Report results ======================

col_w = 35
bar_width = 20  # higher resolution bars
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
