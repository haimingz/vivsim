"""
Benchmark script for profiling LBM function performance.

This script measures the execution time of each core LBM function and reports
the proportion of total runtime consumed by each operation, helping identify
performance bottlenecks.

Usage:
    python examples/benchmark.py

The script benchmarks functions for both a small domain (512x512) and a large
domain (1024x1024) to illustrate how performance characteristics scale.
"""

import time

import jax
import jax.numpy as jnp

from vivsim import lbm


# ========================== CONFIGURATION ==========================

N_WARMUP = 5       # JIT warm-up repetitions (results discarded)
N_REPEAT = 50      # measurement repetitions (results averaged)

DOMAINS = {
    "512x512":   (512,  512),
    "1024x1024": (1024, 1024),
}

# ========================== HELPER ==========================

def timeit(fn, *args, n_warmup=N_WARMUP, n_repeat=N_REPEAT):
    """Return the average wall-clock time (seconds) of a JAX function.

    JAX operations are asynchronous by default; ``block_until_ready`` ensures
    we measure *actual* compute time rather than just dispatch time.
    """
    for _ in range(n_warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    t0 = time.perf_counter()
    for _ in range(n_repeat):
        out = fn(*args)
        jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n_repeat


# ========================== BENCHMARK ==========================

def run_benchmark(nx, ny):
    """Benchmark all core LBM operations for a domain of size (nx, ny)."""

    # ----- shared parameters -----
    nu = 0.01
    omega = lbm.get_omega(nu)
    u0 = 0.1

    # ----- initial arrays -----
    rho = jnp.ones((nx, ny))
    u   = jnp.zeros((2, nx, ny))
    f   = lbm.get_equilibrium(rho, u)
    feq = lbm.get_equilibrium(rho, u)

    # ----- functions to benchmark -----
    benchmarks = [
        ("get_macroscopic",   lambda: lbm.get_macroscopic(f)),
        ("get_equilibrium",   lambda: lbm.get_equilibrium(rho, u)),
        ("collision_bgk",     lambda: lbm.collision_bgk(f, feq, omega)),
        ("streaming",         lambda: lbm.streaming(f)),
        ("boundary_nee (top)",    lambda: lbm.boundary_nee(f, loc="top",    ux_wall=u0)),
        ("boundary_nee (left)",   lambda: lbm.boundary_nee(f, loc="left")),
        ("boundary_nee (right)",  lambda: lbm.boundary_nee(f, loc="right")),
        ("boundary_nee (bottom)", lambda: lbm.boundary_nee(f, loc="bottom")),
    ]

    # ----- measure -----
    results = []
    for name, fn in benchmarks:
        t = timeit(fn)
        results.append((name, t))

    total = sum(t for _, t in results)

    # ----- report -----
    print(f"\n{'─' * 60}")
    print(f"  Domain: {nx} x {ny}   (warmup={N_WARMUP}, repeat={N_REPEAT})")
    print(f"{'─' * 60}")
    print(f"  {'Function':<30} {'Time (ms)':>10}  {'Share':>7}")
    print(f"  {'─'*30} {'─'*10}  {'─'*7}")
    for name, t in results:
        pct = t / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:<30} {t * 1e3:>10.3f}  {pct:>6.1f}%  {bar}")
    print(f"  {'─'*30} {'─'*10}  {'─'*7}")
    print(f"  {'Total (sum of above)':<30} {total * 1e3:>10.3f}  {'100.0%':>7}")
    print()

    # ----- identify bottleneck -----
    bottleneck_name, bottleneck_time = max(results, key=lambda x: x[1])
    print(f"  ⚠  Bottleneck: {bottleneck_name}"
          f"  ({bottleneck_time * 1e3:.3f} ms,"
          f" {bottleneck_time / total * 100:.1f}% of total)")
    print()

    return results


# ========================== MAIN ==========================

if __name__ == "__main__":
    print("\nVivsim LBM Function Benchmark")
    print("=" * 60)
    print(f"  JAX backend : {jax.default_backend()}")
    print(f"  Devices     : {jax.devices()}")

    for label, (nx, ny) in DOMAINS.items():
        run_benchmark(nx, ny)
