"""
Simulate a 2D wedge water entry using IB-LBM coupled with the Shan-Chen multiphase model.
A wedge drops into a pool of liquid.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from tqdm import tqdm
from vivsim import lbm, ib, dyn
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
GRAVITY = 0.0001 # Gravity acceleration

# Initial interface
y0 = NY / 2

# Densities for G = -5.0 in Shan-Chen model
rho_l = 2.4  # Liquid density
rho_g = 0.12 # Gas density

# Wedge geometry
WEDGE_WIDTH = 40
WEDGE_HEIGHT = 30
WEDGE_X = NX / 2
WEDGE_Y = y0 + 10  # Start slightly above water
N_MARKER = 60

# Define a V-shape wedge (tip pointing down)
theta = jnp.linspace(0, 1, N_MARKER // 2)
# Right side
x_r = WEDGE_X + theta * WEDGE_WIDTH / 2
y_r = WEDGE_Y + theta * WEDGE_HEIGHT
# Left side
x_l = WEDGE_X - theta * WEDGE_WIDTH / 2
y_l = WEDGE_Y + theta * WEDGE_HEIGHT
MARKER_X = jnp.concatenate([x_r, x_l[::-1]])
MARKER_Y = jnp.concatenate([y_r, y_l[::-1]])
N_MARKER = len(MARKER_X)
MARKER_DS = jnp.sqrt((MARKER_X[1] - MARKER_X[0])**2 + (MARKER_Y[1] - MARKER_Y[0])**2)

# Wedge kinematics
wedge_v_y = -0.01 # Prescribed falling velocity

def main():
    print("Setting up Wedge entry simulation...")

    # Initialize phase interface
    X, Y = jnp.meshgrid(jnp.arange(NX), jnp.arange(NY), indexing='ij')
    # Use a slightly smoothed interface for better initial stability
    W = 2.0
    rho = rho_g + (rho_l - rho_g) / 2 * (1 - jnp.tanh(2 * (Y - y0) / W))

    u = jnp.zeros((2, NX, NY))
    f = lbm.get_equilibrium(rho, u)

    d = jnp.zeros(2) # Initial displacement
    v = jnp.zeros(2)
    v = v.at[1].set(wedge_v_y)
    a = jnp.zeros(2)

    @jax.jit
    def update(f, d, v, a):
        rho, u = lbm.get_macroscopic(f)

        # 1. Multiphase Interaction Force
        sc_force = get_shan_chen_force(rho, G)

        # 2. Gravity Force
        gravity_force = jnp.zeros((2, NX, NY))
        gravity_force = gravity_force.at[1].set(-GRAVITY * (rho - rho_g))

        # 3. Immersed Boundary (IB) Force
        # Update markers based on current displacement
        marker_x, marker_y = dyn.get_markers_coords_2dof(MARKER_X, MARKER_Y, d)

        # Clip markers to stay within domain boundaries for IB stencil
        marker_x = jnp.clip(marker_x, 2, NX-3)
        marker_y = jnp.clip(marker_y, 2, NY-3)

        stencil_weights, stencil_indices = ib.get_ib_stencil(
            marker_x=marker_x,
            marker_y=marker_y,
            ny=NY,
            kernel=ib.kernel_peskin_4pt,
            stencil_radius=2,
        )
        marker_v = jnp.repeat(v[None, :], N_MARKER, axis=0)

        ib_g, marker_h = ib.multi_direct_forcing(
            grid_u=u,
            stencil_weights=stencil_weights,
            stencil_indices=stencil_indices,
            marker_u_target=marker_v,
            marker_ds=MARKER_DS,
            n_iter=1,
        )

        # Total force = Shan-Chen + Gravity + Immersed Boundary
        force = sc_force + gravity_force + ib_g

        # Shift equilibrium velocity
        u_eq = u + force / rho / OMEGA

        feq = lbm.get_equilibrium(rho, u_eq)
        f = lbm.collision_bgk(f, feq, OMEGA)

        f_before_stream = f
        f = lbm.streaming(f)

        # Boundary conditions
        f = lbm.boundary_bounce_back(f_before_stream, f, loc='bottom')
        f = lbm.boundary_bounce_back(f_before_stream, f, loc='top')
        f = lbm.boundary_bounce_back(f_before_stream, f, loc='left')
        f = lbm.boundary_bounce_back(f_before_stream, f, loc='right')

        # Update wedge position (prescribed constant velocity)
        d = d + v

        return f, d, v, a

    @jax.jit
    def get_density_image(f):
        rho, _ = lbm.get_macroscopic(f)
        return rho.T

    # ========================== CHUNK RUNNER ==========================
    def run_chunk(carry, n_steps):
        def step(carry, _):
            f, d, v, a = carry
            f_next, d_next, v_next, a_next = update(f, d, v, a)
            return (f_next, d_next, v_next, a_next), None
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

        marker_x, marker_y = dyn.get_markers_coords_2dof(MARKER_X, MARKER_Y, d)
        wedge_line, = ax.plot(marker_x, marker_y, 'r-', linewidth=2, label='Wedge')

        plt.colorbar(im, label="Density")
        ax.set_title("Wedge Entry Simulation")
        plt.legend()
        plt.tight_layout()

    # ========================== SIMULATION LOOP ==========================
    chunk_sizes = [CHUNK_STEPS] * (TM // CHUNK_STEPS)
    if TM % CHUNK_STEPS:
        chunk_sizes.append(TM % CHUNK_STEPS)

    print("Starting simulation loop...")

    with tqdm(total=TM, unit="step") as pbar:
        carry = (f, d, v, a)
        for n_steps in chunk_sizes:
            carry, _ = run_chunk_jit(carry, n_steps)
            jax.block_until_ready(carry[0])

            pbar.update(n_steps)

            if PLOT:
                f, d, _, _ = carry
                im.set_data(get_density_image(f))
                marker_x, marker_y = dyn.get_markers_coords_2dof(MARKER_X, MARKER_Y, d)
                wedge_line.set_data(marker_x, marker_y)
                plt.pause(0.001)

    if PLOT:
        plt.ioff()
        plt.close(fig)

    f, d, _, _ = carry

    # Visualize the final density field statically
    rho, _ = lbm.get_macroscopic(f)
    plt.figure(figsize=(8, 4))
    plt.imshow(rho.T, origin='lower', cmap='Blues', vmin=rho_g, vmax=rho_l)

    # Plot final wedge position
    marker_x, marker_y = dyn.get_markers_coords_2dof(MARKER_X, MARKER_Y, d)
    plt.plot(marker_x, marker_y, 'r-', linewidth=2, label='Wedge')

    plt.colorbar(label='Density')
    plt.title(f"Wedge Entry (t={TM})")
    plt.legend()
    plt.tight_layout()

    os.makedirs("output", exist_ok=True)
    plt.savefig("output/wedge_entry.png", dpi=150)
    print("Plot saved to output/wedge_entry.png")

if __name__ == "__main__":
    main()
