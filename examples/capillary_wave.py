"""
Simulate capillary wave decay of a two-phase interface using the Shan-Chen multiphase LBM.
The simulation compares the LBM results with the theoretical damping envelope for
small amplitude capillary waves.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import time
from vivsim import lbm
from vivsim.lbm.multiphase.shan_chen import get_shan_chen_force

# Simulation parameters
NX = 200        # Domain size x
NY = 200        # Domain size y
TM = 20000      # Total time steps
G = -5.0        # Interaction strength
NU = 1/6        # Kinematic viscosity, gives tau = 1.0 (OMEGA = 1.0)
OMEGA = lbm.get_omega(NU)

# Wave parameters
k = 2 * jnp.pi / NX  # Wave number
epsilon = 5.0        # Initial wave amplitude
y0 = NY / 2          # Interface mean height

# Approximate densities for G = -5.0 in Shan-Chen model
rho_l = 2.4  # Liquid density
rho_g = 0.12 # Gas density

def main():
    print("Setting up Capillary Wave simulation...")

    # Initialize phase interface with a sinusoidal perturbation
    X, Y = jnp.meshgrid(jnp.arange(NX), jnp.arange(NY), indexing='ij')
    y_interface = y0 + epsilon * jnp.sin(k * X)

    # Initialize density field (sharp interface, will quickly diffuse to a steady thickness)
    rho = jnp.where(Y < y_interface, rho_l, rho_g)

    u = jnp.zeros((2, NX, NY))
    f = lbm.get_equilibrium(rho, u)

    @jax.jit
    def update(f):
        rho, u = lbm.get_macroscopic(f)

        # Calculate interaction force
        force = get_shan_chen_force(rho, G)

        # Shift equilibrium velocity
        u_eq = u + force / rho / OMEGA

        feq = lbm.get_equilibrium(rho, u_eq)
        f = lbm.collision_bgk(f, feq, OMEGA)
        f = lbm.streaming(f)
        return f

    @jax.jit
    def measure_amplitude(f):
        # Measure interface position using the center of mass in y-direction
        # for a specific column, relative to the mean height.
        rho, _ = lbm.get_macroscopic(f)

        # We track the wave crest at x = NX / 4 (since sin(k * X) peaks at x = lambda/4)
        col_idx = NX // 4
        rho_col = rho[col_idx, :]
        Y_col = jnp.arange(NY)

        # Calculate center of mass in the vertical direction
        # Adjust weight slightly to better track the interface region
        y_c = jnp.sum(rho_col * Y_col) / jnp.sum(rho_col)

        return y_c

    # Run simulation
    amplitudes = []
    times = []

    print("Starting simulation loop...")
    start_time = time.time()

    # Pre-run to let the interface width stabilize before tracking amplitude
    print("Stabilizing interface profile...")
    for t in range(1000):
        f = update(f)

    # Re-normalize our baseline height after stabilization
    baseline_h = measure_amplitude(f).item() - epsilon

    print("Tracking wave decay...")
    for t in range(1000, TM):
        f = update(f)

        if t % 50 == 0:
            h = measure_amplitude(f).item()
            amplitudes.append(h - baseline_h)
            times.append(t)

        if t % 2000 == 0:
            print(f"Step {t}/{TM}, Amplitude: {amplitudes[-1]:.4f}")

    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

    # Approximate theoretical envelope (viscous decay)
    times = jnp.array(times)
    amplitudes = jnp.array(amplitudes)

    # Normalize amplitudes around 0 for better plotting of envelope
    # The shifting is due to density profile asymmetry
    mean_amp = jnp.mean(amplitudes[-100:])
    amplitudes_normalized = amplitudes - mean_amp

    # Simplified theoretical damping rate
    # gamma = 2 * nu * k^2
    nu_eff = NU
    gamma = 2 * nu_eff * (k**2)
    theoretical_decay = amplitudes_normalized[0] * jnp.exp(-gamma * (times - times[0]))

    os.makedirs("output", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(times, amplitudes_normalized, label="LBM Simulation", linewidth=2)
    plt.plot(times, theoretical_decay, 'r--', label="Theoretical Decay Envelope", linewidth=2)
    plt.title("Capillary Wave Decay")
    plt.xlabel("Time steps")
    plt.ylabel("Wave Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/capillary_wave.png", dpi=150)
    print("Plot saved to output/capillary_wave.png")

if __name__ == "__main__":
    main()
