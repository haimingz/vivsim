"""
Simulate a 2D dam break using the Shan-Chen multiphase LBM.
A column of liquid collapses under the influence of gravity.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from tqdm import tqdm
from vivsim import lbm
from vivsim.lbm.multiphase.shan_chen import get_shan_chen_force

# ========================== VISUALIZATION PARAMETERS ====================
PLOT = True   # Enable chunked live plotting
CHUNK_STEPS = 50

# ========================== SIMULATION PARAMETERS ==========================
NX = 200        # Domain size x
NY = 100        # Domain size y
TM = 3000       # Total time steps
G = -5.0        # Interaction strength
NU = 1/6        # Kinematic viscosity, gives tau = 1.0 (OMEGA = 1.0)
OMEGA = lbm.get_omega(NU)
GRAVITY = 0.0001 # Gravity acceleration acting downwards (y-direction)

# Approximate densities for G = -5.0 in Shan-Chen model
rho_l = 2.4  # Liquid density
rho_g = 0.12 # Gas density

def main():
    print("Setting up Dam Break simulation...")

    # Initialize liquid column (dam) on the left side
    dam_width = NX // 4
    dam_height = int(NY * 0.6)

    X, Y = jnp.meshgrid(jnp.arange(NX), jnp.arange(NY), indexing='ij')

    # Initialize density field: liquid inside the dam region, gas everywhere else
    rho = jnp.where((X < dam_width) & (Y < dam_height), rho_l, rho_g)

    u = jnp.zeros((2, NX, NY))
    f = lbm.get_equilibrium(rho, u)

    @jax.jit
    def update(f):
        rho, u = lbm.get_macroscopic(f)

        # Calculate multiphase interaction force
        sc_force = get_shan_chen_force(rho, G)

        # Calculate gravity force: F_g = g * (rho - rho_g)
        # We subtract gas density to mitigate spurious acceleration in the lighter phase
        gravity_force = jnp.zeros((2, NX, NY))
        gravity_force = gravity_force.at[1].set(-GRAVITY * (rho - rho_g))

        # Total force
        force = sc_force + gravity_force

        # Exact Difference Method (EDM) or Guo forcing shift
        u_eq = u + force / rho / OMEGA

        feq = lbm.get_equilibrium(rho, u_eq)
        f = lbm.collision_bgk(f, feq, OMEGA)

        f_before_stream = f
        f = lbm.streaming(f)

        # Solid boundaries: apply bounce back on all walls (bottom, top, left, right)
        f = lbm.boundary_bounce_back(f_before_stream, f, loc='bottom')
        f = lbm.boundary_bounce_back(f_before_stream, f, loc='top')
        f = lbm.boundary_bounce_back(f_before_stream, f, loc='left')
        f = lbm.boundary_bounce_back(f_before_stream, f, loc='right')

        return f

    @jax.jit
    def get_density_image(f):
        rho, _ = lbm.get_macroscopic(f)
        return rho.T

    # ========================== CHUNK RUNNER ==========================
    def run_chunk(carry, n_steps):
        def step(f, _):
            f_next = update(f)
            return f_next, None
        return jax.lax.scan(step, carry, None, length=n_steps)

    run_chunk_jit = jax.jit(run_chunk, static_argnums=1)

    # ======================= VISUALIZATION SETUP ====================
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
        ax.set_title("Dam Break Simulation")
        plt.tight_layout()

    # ========================== SIMULATION LOOP ==========================
    chunk_sizes = [CHUNK_STEPS] * (TM // CHUNK_STEPS)
    if TM % CHUNK_STEPS:
        chunk_sizes.append(TM % CHUNK_STEPS)

    print("Starting simulation loop...")

    with tqdm(total=TM, unit="step") as pbar:
        for n_steps in chunk_sizes:
            f, _ = run_chunk_jit(f, n_steps)
            jax.block_until_ready(f)

            pbar.update(n_steps)

            if PLOT:
                im.set_data(get_density_image(f))
                plt.pause(0.001)

    if PLOT:
        plt.ioff()
        plt.close(fig)

    # Visualize the final density field statically
    rho, u = lbm.get_macroscopic(f)
    plt.figure(figsize=(8, 4))
    plt.imshow(rho.T, origin='lower', cmap='Blues', vmin=rho_g, vmax=rho_l)
    plt.colorbar(label='Density')
    plt.title(f"Dam Break (t={TM})")
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/dam_break.png", dpi=150)
    print("Simulation complete. Plot saved to output/dam_break.png")

if __name__ == "__main__":
    main()
