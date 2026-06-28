"""
Vector Form Intrinsic Finite Element (VFIFE) for a flexible beam

This example simulates the vibration of a flexible cantilever beam
using the VFIFE method.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from vivsim import dyn

# ========================== GEOMETRY & MATERIAL =======================
L = 10.0           # Length of the beam
N_NODES = 20       # Number of nodes
N_ELEM = N_NODES - 1

EA = 1e6           # Axial stiffness
EI = 1e4           # Bending stiffness
RHO_L = 1.0        # Mass per unit length
M_NODE = RHO_L * L / N_ELEM  # Mass per node
J_NODE = M_NODE * (L / N_ELEM)**2 / 12  # Moment of inertia per node

DAMPING = 0.5      # Damping coefficient

# Initial positions
x0 = jnp.linspace(0, L, N_NODES)
y0 = jnp.zeros(N_NODES)
x = jnp.stack([x0, y0], axis=-1)

l0 = jnp.ones(N_ELEM) * (L / N_ELEM)

# State variables
v = jnp.zeros((N_NODES, 2))
theta = jnp.zeros(N_NODES)
omega = jnp.zeros(N_NODES)

# Mass and moment of inertia arrays
m = jnp.ones(N_NODES) * M_NODE
J = jnp.ones(N_NODES) * J_NODE

# External forces
f_ext = jnp.zeros((N_NODES, 2))
m_ext = jnp.zeros(N_NODES)

# Apply initial displacement (e.g., pulling the tip)
x = x.at[-1, 1].set(-2.0)

# ========================== SIMULATION PARAMETERS =======================
DT = 1e-4
TM = 200000
PLOT_FREQ = 2000

# ========================== SIMULATION ROUTINE =====================
@jax.jit
def update(x, v, theta, omega):
    x_next, v_next, theta_next, omega_next = dyn.vfife_beam(
        x, v, theta, omega, f_ext, m_ext, m, J, EA, EI, l0, DT, DAMPING
    )

    # Boundary condition: fixed at left end (cantilever)
    x_next = x_next.at[0].set(x[0])
    v_next = v_next.at[0].set(0.0)
    theta_next = theta_next.at[0].set(0.0)
    omega_next = omega_next.at[0].set(0.0)

    return x_next, v_next, theta_next, omega_next

# ========================== RUN SIMULATION ==========================
history_x = []
history_y = []

# Using scan for faster execution
def run_chunk(carry, n_steps):
    def step(carry, _):
        x, v, theta, omega = carry
        carry_next = update(x, v, theta, omega)
        return carry_next, carry_next[0]  # Return x as output
    return jax.lax.scan(step, carry, None, length=n_steps)

run_chunk = jax.jit(run_chunk, static_argnums=1)

carry = (x, v, theta, omega)

# Run warmup
carry, _ = run_chunk(carry, 10)

print(f"Running simulation for {TM} steps...")
with tqdm(total=TM) as pbar:
    for i in range(TM // PLOT_FREQ):
        carry, x_chunk = run_chunk(carry, PLOT_FREQ)
        jax.block_until_ready(x_chunk)

        # Save end position for plotting
        x_last = x_chunk[-1]
        history_x.append(x_last[:, 0])
        history_y.append(x_last[:, 1])

        pbar.update(PLOT_FREQ)

print("Plotting results...")
plt.figure(figsize=(10, 4))
for i, (hx, hy) in enumerate(zip(history_x, history_y)):
    if i % 10 == 0:  # plot every 10th saved frame
        plt.plot(hx, hy, 'b-', alpha=float(i)/len(history_x)*0.8+0.2)

plt.plot(history_x[-1], history_y[-1], 'r-', linewidth=2, label='Final state')
plt.plot(x0, y0, 'k--', label='Initial undeformed')
plt.plot(x[:, 0], x[:, 1], 'g:', label='Initial deformed')

plt.xlim(-1, L + 1)
plt.ylim(-3, 3)
plt.grid(True)
plt.legend()
plt.title("VFIFE Cantilever Beam Vibration")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("vfife_beam_result.png")
print("Saved plot to vfife_beam_result.png")
