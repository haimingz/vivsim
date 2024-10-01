<p align="center">
<img src ="assets/vivsim.svg"/>
</p>

# VIVSIM 

VIVSIM is a Python library for accelerated fluid-structure interaction (FSI) simulations based on the immersed boundary -lattice Boltzmann method (IB-LBM). It was originated from a research project requiring efficient simulation codes for studying vortex-induced vibration (VIV) of underwater structures. 

Similar to projects like [JAX-CFD](https://github.com/google/jax-cfd) and [XLB](https://github.com/Autodesk/XLB), VIVSIM utilizes [JAX](https://github.com/jax-ml/jax) as the backend to harness the power of hardware accelerators (mainly GPUs) with little extra efforts.

## Examples

### Fluid flow without interaction with objects

<p align="center">
    <img src ="assets/cavity.gif" width=500/>
</p>

> Simulation of lid-driven cavity at Re = 20000 on a 1000x1000 lattice grid.

<p align="center">
    <img src ="assets/text.gif" width=500/>
</p>

> Simulation of flow pass through objects (texts) on a 1000x1000 lattice grid.

### Fluid-Structure Interaction (FSI)

<p align="center">
    <img src ="assets/viv_1000.gif" width=800/>
</p>

> Vorticity contour showing the VIV of a cylinder with Vr = 5 and Re = 1000. The simulation was conducted on a 800x400 lattice grid with 100 immersed boundary markers, taking about 1 min on Nvidia Geforce GTX 1080.

<p align="center">
    <img src ="assets/viv_10000.gif" width=800/>
</p>

> Velocity and vorticity contours showing the VIV of a cylinder with Vr = 5 and Re = 10000. The simulation was conducted on a 8000x4000 lattice grid with 1200 immersed boundary markers, taking about 30 min on 8 Nvidia A800.

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

This package is based on JAX, whose installation may depend on the OS and hardware. Please refer to the [JAX Documentation](https://jax.readthedocs.io/en/latest/installation.html) for the latest installation guidance. 

To locally install VIVSIM for development:

```bash
git clone https://github.com/haimingz/vivsim.git
cd vivsim
pip install -r requirements.txt
pip install -e .
```

Also, you can create and run the following cell to install VIVSIM on Google Collab:

```python
!pip install git+https://github.com/haimingz/vivsim
```

