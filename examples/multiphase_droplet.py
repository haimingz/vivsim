"""
Simulate phase separation of a single component fluid using the Shan-Chen multiphase LBM.
A droplet is formed from an initially random density distribution.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from vivsim import lbm

# Set up parameters
NX = 100
NY = 100
TM = 4000
G = -5.0     # Interaction strength (negative for attraction)
OMEGA = 1.0  # relaxation time tau = 1.0 -> nu = 1/6

def main():
    print(f"Initializing Shan-Chen multiphase simulation on a {NX}x{NY} grid.")
    print(f"G = {G}, OMEGA = {OMEGA}, Total steps = {TM}")

    # Initialize density with random noise to trigger phase separation
    rho = jnp.ones((NX, NY))
    key = jax.random.PRNGKey(42)
    noise = jax.random.uniform(key, shape=(NX, NY), minval=-0.01, maxval=0.01)
    rho = rho + noise

    u = jnp.zeros((2, NX, NY))
    f = lbm.get_equilibrium(rho, u)

    @jax.jit
    def update(f):
        rho, u = lbm.get_macroscopic(f)

        # Calculate Shan-Chen interaction force
        force = lbm.get_shan_chen_force(rho, G)

        # In Shan-Chen model, the equilibrium velocity is shifted by force
        u_eq = u + force / rho / OMEGA

        # Compute equilibrium
        feq = lbm.get_equilibrium(rho, u_eq)

        # BGK collision
        f = lbm.collision_bgk(f, feq, OMEGA)

        # Streaming
        f = lbm.streaming(f)
        return f

    # Run simulation
    for t in range(TM):
        f = update(f)
        if (t + 1) % 500 == 0:
            rho, _ = lbm.get_macroscopic(f)
            print(f"Step {t + 1}, min rho: {jnp.min(rho):.4f}, max rho: {jnp.max(rho):.4f}")

    # Visualize the final density field
    rho, u = lbm.get_macroscopic(f)
    plt.figure(figsize=(6, 5))
    plt.imshow(rho.T, origin='lower', cmap='viridis')
    plt.colorbar(label='Density')
    plt.title(f"Shan-Chen Phase Separation (t={TM})")
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/multiphase_droplet.png")
    print("Simulation complete. Output saved to output/multiphase_droplet.png")

if __name__ == "__main__":
    main()
