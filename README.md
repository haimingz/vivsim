<br/>
<p align="center">
<img src ="assets/vivsim.svg"/><br/><br/>
</p>

## VIVSIM 

[![GitHub Traffic](https://img.shields.io/badge/dynamic/json?color=success&label=Views&query=count&url=https://gist.githubusercontent.com/haimingz/b2e30dd706e57413c1c1f688de82ef6e/raw/traffic.json&logo=github)](https://github.com/MShawon/github-clone-count-badge) [![GitHub Clones](https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count&url=https://gist.githubusercontent.com/haimingz/8981033dc17c32a7c2f409e631b57309/raw/clone.json&logo=github)](https://github.com/MShawon/github-clone-count-badge)


VIVSIM is a Python library for fluid-structure interaction (FSI) simulations based on the immersed boundary -lattice Boltzmann method (IB-LBM). It was originated from a research project requiring efficient simulation codes for studying vortex-induced vibration (VIV) of underwater structures. 

Inspired by projects like [JAX-CFD](https://github.com/google/jax-cfd) and [XLB](https://github.com/Autodesk/XLB), VIVSIM utilizes [JAX](https://github.com/jax-ml/jax) as the backend to achieve *hardware acceleration* and *automatic differentiation*. The project follows the **Functional Programming** paradigm to facilitate XLA compilation while making the codebase easier to understand and maintain.

<div align="center">
  <table style="border: none; text-align: center;">
    <tr>
      <td><i>Lid-driven cavity at Re = 2e4 </i></td>
      <td><i>Flow passes some texts </i></td>
    </tr>
    <tr>
      <td><img src="assets/cavity.gif" alt="Lid-driven cavity flow" width="250"></td>
      <td><img src="assets/text.gif" alt="Flow past text" width="250"></td>
    </tr>
    <tr>
      <td><i>VIV of a cylinder with U_r = 5 and Re = 1e2</i></td>
      <td><i>VIV of a cylinder with U_r = 5 and Re = 1e4</i></td>
    </tr>
    <tr>
      <td><img src="assets/viv_100.gif" alt="VIV at Re = 1e2" width="300"></td>
      <td><img src="assets/viv_10000.gif" alt="VIV at Re = 1e4" width="270"></td>
    </tr>
  </table>
</div>

## Installation

To locally install VIVSIM for development:

```bash
git clone https://github.com/haimingz/vivsim.git
cd vivsim
pip install -e ".[cpu]"
```

JAX installation depends on the operating system and accelerator backend. VIVSIM now exposes the most common JAX choices as optional extras:

```bash
# CPU-only development on Linux/macOS/Windows
pip install -e ".[cpu]"

# NVIDIA GPU on Linux
pip install -e ".[cuda12]"
pip install -e ".[cuda13]"

# Google Cloud TPU
pip install -e ".[tpu]"
```

## Usage

VIVSIM provides a collection of **pure functions** for IB-LBM computations. Users can construct custom simulation models for different tasks. Start with the included demo examples to see how easy that is! Below is a minimum workable example for lid-driven cavity simulation:

```python
import jax
import jax.numpy as jnp
from vivsim import lbm

# define constants
NX = 100  # grid size in x direction
NY = 100  # grid size in y direction
TM = 1000  # number of time steps
U0 = 0.5  # velocity of lid
NU = 0.1  # kinematic viscosity

OMEGA = lbm.get_omega(NU)  # relaxation parameter

# define fluid properties
rho = jnp.ones((NX, NY), dtype=jnp.float32)      # density
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)    # velocity

# initialize distribution function
f = lbm.get_equilibrium(rho, u)

# define compute routine
@jax.jit
def update(f):
    
    # Collision
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_bgk(f, feq, OMEGA)

    # Streaming
    f = lbm.streaming(f)

    # Boundary conditions
    f = lbm.boundary_nee(f, loc='left')
    f = lbm.boundary_nee(f, loc='right')
    f = lbm.boundary_nee(f, loc='bottom')
    f = lbm.boundary_nee(f, loc='top', ux_wall=U0)
    
    return f

# start simulation
for t in range(TM):   
    f  = update(f)

    # export data & visualization ...

```

## Implemented Methods

Lattice Models
- D2Q9
  
Collision Models
- Bhatnagar-Gross-Krook (BGK) collision operator
- Multiple Relaxation Time (MRT) collision operator
- Karlin–Bösch–Chikatamarla (KBC) collision operator
- Regularized collision operator

Boundary Conditions:
- Predescribed velocity, density, and forces at boundaies using: 
  - Non-Equilibrium Bounce Back (NEBB) method
  - Non-Equilibrium Extrapolation (NEE) method
  - Equilibrium boundary
- Predescribed velocity boundary using
  - Bounce-Back method (no-slip)
  - Specular Reflection (slip)
- Periodic boundary

Forcing Schemes:
- Guo's Forcing scheme
- Modified Exact Difference Method (EDM)

Immersed Boundary Methods:
- Peskin's 2-, 3- and 4-point kernels
- 4-point cosine kernel
- Multi-Direct-Forcing (MDF) method.

## Benchmark 

Here is a performance snippet showing the average execution time (ms) for core JAX-compiled pure functions on a Nvidia RTX4090 GPU. You can generate similar benchmarks on your hardware by running `python examples/benchmark.py`.

```
Benchmarking on cuda:0 | grid: 1000x1000 | repeats: 100

Function                      Time (ms)  Bar
-----------------------------------------------------------------------------------
lbm.get_macroscopic               0.002  
lbm.get_equilibrium               0.001  
lbm.streaming                     0.066  =========
lbm.collision_bgk                 0.030  ====
lbm.collision_kbc                 0.156  =======================
lbm.collision_mrt                 0.264  ========================================
lbm.collision_regularized         0.087  =============
lbm.forcing_edm                   0.105  ===============
lbm.forcing_guo_bgk               0.060  =========
lbm.forcing_guo_mrt               0.219  =================================
lbm.get_guo_forcing_term          0.001  
lbm.boundary_nee                  0.007  =
lbm.boundary_velocity_nee         0.008  =
lbm.boundary_pressure_nee         0.008  =
lbm.boundary_nebb                 0.014  ==
lbm.boundary_velocity_nebb        0.011  =
lbm.boundary_pressure_nebb        0.069  ==========
lbm.boundary_equilibrium          0.005  
lbm.boundary_bounce_back          0.118  =================
lbm.boundary_specular_reflection  0.124  ==================
lbm.obstacle_bounce_back          0.001  
lbm.boundary_characteristic       0.001  
ib.get_area                       0.001  
ib.get_ds_closed                  0.001  
ib.get_ds_open                    0.001  
ib.kernel_peskin_3pt              0.004  
ib.kernel_peskin_4pt              0.003  
ib.kernel_cosine_4pt              0.004  
ib.get_ib_stencil                 0.001  
ib.interpolate                    0.001  
ib.spread                         0.001  
ib.multi_direct_forcing           0.001  
```
If you have suggestions for improving the performance of any function, please feel free to open an issue or submit a pull request!


## Cite VIVSIM

If you find this repo useful, please cite [our paper](https://asmedigitalcollection.asme.org/OMAE/proceedings-abstract/OMAE2024/87844/1202724):
```
@article{zhu2025gpu,
  title={{GPU} Accelerated Vortex-Induced Vibration Simulation Using {JAX}: Efficiency and Accuracy Strategies},
  author={Zhu, Haiming and Yang, Yuan and Du, Zunfeng and Yu, Jianxing},
  journal={Computers \& Fluids},
  pages={106913},
  year={2025},
  publisher={Elsevier}
}
```
