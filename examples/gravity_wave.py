"""
Simulate a 2D gravity wave using the Shan-Chen multiphase LBM.
The simulation compares the LBM wave frequency with the theoretical solution
for a standing gravity wave.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from vivsim import lbm
from vivsim.lbm.multiphase.shan_chen import get_shan_chen_force

# ========================== VISUALIZATION PARAMETERS ====================
PLOT = True   # Enable chunked live plotting
CHUNK_STEPS = 50

# ========================== SIMULATION PARAMETERS ==========================
NX = 200        # Domain size x
NY = 100        # Domain size y
TM = 10000      # Total time steps
G = -5.0        # Interaction strength
NU = 1/6        # Kinematic viscosity, gives tau = 1.0 (OMEGA = 1.0)
OMEGA = lbm.get_omega(NU)
GRAVITY = 0.0001 # Gravity acceleration

# Wave parameters
k = 2 * jnp.pi / NX  # Wave number
epsilon = 2.0        # Initial wave amplitude
y0 = NY / 2          # Interface mean height

# Approximate densities for G = -5.0 in Shan-Chen model
rho_l = 2.4  # Liquid density
rho_g = 0.12 # Gas density

def main():
    print("Setting up Gravity Wave simulation...")

    # Initialize phase interface with a sinusoidal perturbation
    X, Y = jnp.meshgrid(jnp.arange(NX), jnp.arange(NY), indexing='ij')
    # Smooth initial interface to reduce start-up oscillations
    W = 2.0
    y_interface = y0 + epsilon * jnp.cos(k * X)
    rho = rho_g + (rho_l - rho_g) / 2 * (1 - jnp.tanh(2 * (Y - y_interface) / W))

    u = jnp.zeros((2, NX, NY))
    f = lbm.get_equilibrium(rho, u)

    @jax.jit
    def update(f):
        rho, u = lbm.get_macroscopic(f)

        # Calculate interaction force
        sc_force = get_shan_chen_force(rho, G)

        # Calculate gravity force
        gravity_force = jnp.zeros((2, NX, NY))
        gravity_force = gravity_force.at[1].set(-GRAVITY * (rho - rho_g))

        force = sc_force + gravity_force

        # Shift equilibrium velocity
        u_eq = u + force / rho / OMEGA

        feq = lbm.get_equilibrium(rho, u_eq)
        f = lbm.collision_bgk(f, feq, OMEGA)

        f_before_stream = f
        f = lbm.streaming(f)

        # Boundary conditions: bounce back on all walls
        f = lbm.boundary_bounce_back(f_before_stream, f, loc='bottom')
        f = lbm.boundary_bounce_back(f_before_stream, f, loc='top')
        f = lbm.boundary_bounce_back(f_before_stream, f, loc='left')
        f = lbm.boundary_bounce_back(f_before_stream, f, loc='right')

        return f

    @jax.jit
    def measure_amplitude(f):
        rho, _ = lbm.get_macroscopic(f)

        # Track the wave crest at x = 0
        col_idx = 0
        rho_col = rho[col_idx, :]
        Y_col = jnp.arange(NY)

        # Track the point where density crosses the mean density
        rho_mean = (rho_l + rho_g) / 2
        # Simple center of mass approach for the interface location
        y_c = jnp.sum(rho_col * Y_col * (rho_col > rho_mean)) / jnp.sum(rho_col * (rho_col > rho_mean))
        return y_c

    @jax.jit
    def get_density_image(f):
        rho, _ = lbm.get_macroscopic(f)
        return rho.T

    # ========================== CHUNK RUNNER ==========================
    def run_chunk(carry, n_steps):
        def step(f, _):
            f_next = update(f)
            amp = measure_amplitude(f_next)
            return f_next, amp
        return jax.lax.scan(step, carry, None, length=n_steps)

    run_chunk_jit = jax.jit(run_chunk, static_argnums=1)

    # ======================= VISUALIZATION SETUP ====================
    import matplotlib as mpl
    mpl.rcParams["figure.raise_window"] = False

    if PLOT:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(
            get_density_image(f),
            extent=[0, NX, 0, NY],
            cmap="Blues", origin="lower",
            vmin=rho_g, vmax=rho_l
        )
        plt.colorbar(im, label="Density")
        ax.set_title("Gravity Wave")
        plt.tight_layout()

    # ========================== SIMULATION LOOP ==========================
    amplitudes = []
    chunk_sizes = [CHUNK_STEPS] * (TM // CHUNK_STEPS)
    if TM % CHUNK_STEPS:
        chunk_sizes.append(TM % CHUNK_STEPS)

    print("Starting simulation loop...")

    # Let interface diffuse initially
    f, _ = run_chunk_jit(f, 500)
    baseline_h = measure_amplitude(f).item() - epsilon

    with tqdm(total=TM, unit="step") as pbar:
        for n_steps in chunk_sizes:
            f, amps_chunk = run_chunk_jit(f, n_steps)
            jax.block_until_ready(f)

            amplitudes.extend(amps_chunk.tolist())
            pbar.update(n_steps)

            if PLOT:
                im.set_data(get_density_image(f))
                plt.pause(0.001)

    if PLOT:
        plt.ioff()
        plt.close(fig)

    # ========================== ANALYSIS ==========================
    amplitudes = jnp.array(amplitudes)
    # Center the amplitudes around 0
    mean_amp = jnp.mean(amplitudes[-2000:])
    amplitudes = amplitudes - mean_amp
    times = jnp.arange(TM)

    # Theoretical dispersion relation for deep water gravity waves
    omega_theoretical = jnp.sqrt(GRAVITY * k * jnp.tanh(k * y0))
    theoretical_wave = epsilon * jnp.cos(omega_theoretical * times) * jnp.exp(-2 * NU * k**2 * times)

    plt.figure(figsize=(8, 5))
    plt.plot(times, amplitudes, label="LBM Simulation", linewidth=2)
    plt.plot(times, theoretical_wave, 'r--', label="Theoretical Wave", linewidth=2)
    plt.title("Gravity Wave Oscillation")
    plt.xlabel("Time steps")
    plt.ylabel("Wave Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    import os
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/gravity_wave_comparison.png", dpi=150)
    print("Plot saved to output/gravity_wave_comparison.png")

if __name__ == "__main__":
    main()
