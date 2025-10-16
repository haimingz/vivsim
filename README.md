<br/>
<p align="center">
<img src ="assets/vivsim.svg"/><br/><br/>
</p>

## VIVSIM 

[![GitHub Traffic](https://img.shields.io/badge/dynamic/json?color=success&label=Views&query=count&url=https://gist.githubusercontent.com/haimingz/b2e30dd706e57413c1c1f688de82ef6e/raw/traffic.json&logo=github)](https://github.com/MShawon/github-clone-count-badge) [![GitHub Clones](https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count&url=https://gist.githubusercontent.com/haimingz/8981033dc17c32a7c2f409e631b57309/raw/clone.json&logo=github)](https://github.com/MShawon/github-clone-count-badge)


VIVSIM is a Python library for accelerated fluid-structure interaction (FSI) simulations based on the immersed boundary -lattice Boltzmann method (IB-LBM). It was originated from a research project requiring efficient simulation codes for studying vortex-induced vibration (VIV) of underwater structures. 

Inspired by projects like [JAX-CFD](https://github.com/google/jax-cfd) and [XLB](https://github.com/Autodesk/XLB), VIVSIM utilizes [JAX](https://github.com/jax-ml/jax) as the backend to harness the power of hardware accelerators, achieving massive parallelism on GPU/GPUs. 



## Recent Updates

**Version 1.1.0** (October 2024)
- **Modular LBM Architecture**: Reorganized all LBM-related code into a structured `lbm` module with submodules for `boundary`, `collision`, and `forcing`, enabling easier maintenance and future extensions with new methods.
- **Enhanced Boundary Conditions**: Implemented more boundary condition methods including Non-Equilibrium Extrapolation (NEE), Non-Equilibrium Bounce Back (NEBB), and equilibrium boundary conditions. Support predescribed velocity, density, and forces.
- **Improved LBM Collision Operators**: Implemented Karlin-Bösch-Chikatamarla (KBC) collision operator that is super stable for high Re numbers.
- **New Forcing Scheme**: Implemented the modified Exact Difference Method (EDM) which requires no correction to the collision operator when there are forces..
- **Code Quality**: Improved code readability, fixed typos in docstrings, and enhanced documentation throughout the codebase using AI with my supervision. 


## Usage

VIVSIM provides a collection of **pure functions** for IB-LBM computations. Users can construct custom simulation models for different tasks. Start with the included demo examples to see how easy that is! 

Below is a minimum workable example for lid-driven cavity simulation:

```python
import jax
import jax.numpy as jnp
from vivsim import lbm

# define constants
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
    
    return f, rho, u

# start simulation
for t in range(TM):   
    f, rho, u  = update(f)

    # export data & visualization ...

```

## Examples

_Lid-driven cavity at Re = 2e4 on a 1000x1000 lattice grid_

<img src="assets/cavity.gif" alt="Lid-driven cavity flow" width="300">  

_Flow passes some texts on a 1000x1000 lattice grid_

<img src="assets/text.gif" alt="Flow past text" width="300">  

_VIV of a cylinder with U_r = 5 and Re = 1e2_

<img src="assets/viv_100.gif" alt="VIV at Re = 1e2" width="300">  

_VIV of a cylinder with U_r = 5 and Re = 1e4_

<img src="assets/viv_10000.gif" alt="VIV at Re = 1e4" width="300">  

## Capabilities

Lattice Models
- D2Q9
  
Collision Models
- Bhatnagar-Gross-Krook (BGK) collision operator
- Multiple Relaxation Time (MRT) collision operator
- Karlin–Bösch–Chikatamarla (KBC) collision operator

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
- Guo's Forcing sheme
- Modified Exact Difference Method (EDM)

Fluid-Structure Interaction
- Multi-Direct-Forcing (MDF) Immersed Boundary method.

Acceleration techniques
- Multi-GPU simulation (using JAX)
- Gird refinement (shown below)
- Dynamic IB region (shown below)

<img src="assets/grid_refinement.png" width=500 />

## Todos

- Standardized simulation routines.
- 3D simulation capability.


## Getting Started

To locally install VIVSIM for development:

```bash
git clone https://github.com/haimingz/vivsim.git
pip install -e vivsim
```
This package is based on JAX, whose installation may depend on the OS and hardware. If the above command does not work well, please refer to the [JAX Documentation](https://jax.readthedocs.io/en/latest/installation.html) for the latest installation guidance. 

More detailed instructions can be found in our [Documentation](https://github.com/haimingz/vivsim/wiki/Installation).

## Cite VIVSIM

If you find this repo useful, please cite [our paper](https://asmedigitalcollection.asme.org/OMAE/proceedings-abstract/OMAE2024/87844/1202724):
```
@inproceedings{zhu2024computational,
  title={Computational Performance of IB-LBM Based VIV Simulation Using Python With JAX},
  author={Zhu, Haiming and Du, Zunfeng and Yang, Yuan and Han, Muxuan},
  booktitle={International Conference on Offshore Mechanics and Arctic Engineering},
  volume={87844},
  pages={V006T08A020},
  year={2024},
  organization={American Society of Mechanical Engineers}
}
```

