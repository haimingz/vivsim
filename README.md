<p align="center">
<img src ="assets/vivsim.svg"/>
</p>

# VIVSIM 

Accelerated vortex-induced vibration (VIV) simulation using immersed boundary lattice Boltzmann method (IB-LBM) powered by JAX.

## Examples

### Fluid flow without interaction with objects

| ![](assets/ldc.gif) | ![](assets/fot.gif) |
| -------- | -------- | 
| Lid-driven cavity (Re = 20000) | Flow over texts (Re=3000) | 

### Fluid-Structure Interaction (FSI)

<p align="center">
    <img src ="assets/viv.gif"/>
</p>
Vorticity contour of the VIV of a cylinder at Re = 1000 with 800x400 cells and 100 immersed boundary markers. The simulation takes about 1 min on Nvidia Geforce 1080.


## Capabilities

Lattice Models
- D2Q9
  
Collision Models
- Bhatnagar-Gross-Krook (BGK) model
- Multiple Relaxation Time (MRT) model

Boundary Conditions:
- Velocity boundary using Non-Equlibrium Bounce Back (Zou-He) method
- No slip boundary using Halfway Bounce-Back method
- Outlet boundary using no gradient method
- Periodic boundary

Fluid-Structure Interaction
- Multi Direct-Forcing Immersed Boundary method
- Confined Immersed Boundary technique to speed up VIV simulation

## Getting Started

To locally install for development:

```bash
git clone https://github.com/haimingz/vivsim.git
cd vivsim
pip install -r requirements.txt
pip install -e .
```

Please refer to https://github.com/google/jax for the latest installation documentation. 