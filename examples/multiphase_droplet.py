"""
Simulate phase separation of a single component fluid using the Shan-Chen multiphase LBM.
A droplet is formed from an initially random density distribution.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from tqdm import tqdm
from vivsim import lbm

# ========================== VISUALIZATION PARAMETERS ====================
PLOT = True   # Enable chunked live plotting
CHUNK_STEPS = 50

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

    @jax.jit
    def get_density_image(f):
        rho, _ = lbm.get_macroscopic(f)
        return rho.T

    # ========================== CHUNK RUNNER ==========================
    def run_chunk(carry, n_steps):
        def step(carry, _):
            f = carry
            f_next = update(f)
            return f_next, None
        return jax.lax.scan(step, carry, None, length=n_steps)

    run_chunk_jit = jax.jit(run_chunk, static_argnums=1)

    # ======================= VISUALIZATION SETUP ====================
    mpl.rcParams["figure.raise_window"] = False

    if PLOT:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            get_density_image(f),
            extent=[0, NX, 0, NY],
            cmap="viridis", origin="lower",
        )
        plt.colorbar(im, label="Density")
        ax.set_title("Shan-Chen Phase Separation")
        plt.tight_layout()

    # ========================== SIMULATION LOOP ==========================
    chunk_sizes = [CHUNK_STEPS] * (TM // CHUNK_STEPS)
    if TM % CHUNK_STEPS:
        chunk_sizes.append(TM % CHUNK_STEPS)

    print("Starting simulation loop...")

    with tqdm(total=TM, unit="step") as pbar:
        carry = f
        for n_steps in chunk_sizes:
            carry, _ = run_chunk_jit(carry, n_steps)
            jax.block_until_ready(carry)

            pbar.update(n_steps)

            if PLOT:
                f_current = carry
                rho_current = get_density_image(f_current)
                im.set_data(rho_current)
                # Note: vmin/vmax update for dynamic range of separation
                im.set_clim(vmin=jnp.min(rho_current), vmax=jnp.max(rho_current))
                plt.pause(0.001)

    if PLOT:
        plt.ioff()
        plt.close(fig)

    f = carry

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
