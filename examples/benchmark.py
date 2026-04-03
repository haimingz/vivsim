"""
Performance benchmark for vivsim LBM functions.

This script measures the execution time of each core LBM function and reports
the time consumption ratio, making it easy to identify performance bottlenecks.

Usage:
    python examples/benchmark.py

Requirements:
    pip install vivsim tqdm
"""

import time

import jax
import jax.numpy as jnp

from vivsim import lbm

# ====================== Configuration ======================

NX = 500          # grid size in x direction
NY = 500          # grid size in y direction
N_WARMUP = 20     # number of warm-up iterations (for JIT compilation)
N_REPEAT = 100    # number of timed iterations for each function

# ====================== Initialize ======================

U0 = 0.1
NU = 0.01
OMEGA = lbm.get_omega(NU)

rho = jnp.ones((NX, NY))
u = jnp.zeros((2, NX, NY))
f = lbm.get_equilibrium(rho, u)
feq = lbm.get_equilibrium(rho, u)

# ====================== Timing helper ======================

def measure(fn, *args):
    """Warm up a function then measure its average wall-clock time (ms)."""
    for _ in range(N_WARMUP):
        result = fn(*args)
        jax.block_until_ready(result)

    start = time.perf_counter()
    for _ in range(N_REPEAT):
        result = fn(*args)
        jax.block_until_ready(result)
    elapsed = time.perf_counter() - start
    return elapsed / N_REPEAT * 1000  # ms per call


# ====================== Benchmark functions ======================

print(f"\nBenchmarking on {jax.devices()[0]} | grid: {NX}x{NY} | repeats: {N_REPEAT}\n")

timings = {}

timings["get_macroscopic"] = measure(lbm.get_macroscopic, f)
timings["get_equilibrium"] = measure(lbm.get_equilibrium, rho, u)
timings["collision_bgk"]   = measure(lbm.collision_bgk, f, feq, OMEGA)
timings["collision_kbc"]   = measure(lbm.collision_kbc, f, feq, OMEGA)
timings["streaming"]       = measure(lbm.streaming, f)


def apply_all_boundaries(f):
    f = lbm.boundary_nee(f, loc="left")
    f = lbm.boundary_nee(f, loc="right")
    f = lbm.boundary_nee(f, loc="bottom")
    f = lbm.boundary_nee(f, loc="top", ux_wall=U0)
    return f


timings["boundary_nee (×4)"] = measure(apply_all_boundaries, f)

# ====================== Report results ======================

total = sum(timings.values())
bar_width = 40

print(f"{'Function':<24} {'Time (ms)':>10}  {'%':>7}  Bar")
print("-" * 70)
for name, t in sorted(timings.items(), key=lambda x: -x[1]):
    pct = t / total * 100
    bar = "█" * int(pct / 100 * bar_width)
    print(f"{name:<24} {t:>10.3f}  {pct:>6.1f}%  {bar}")
print("-" * 70)
print(f"{'Total (one update step)':<24} {total:>10.3f}  {'100.0%':>7}")
print()
