<p align="center">
<img src ="assets/vivsim.svg"/>
</p>

# VIVSIM 

Accelerated vortex-induced vibration (VIV) simulation using immersed boundary lattice Boltzmann method (IB-LBM) powered by JAX.

<p align="center">
    <img src ="assets/viv.gif"/>
</p>

> This is a real-time VIV simulation at Re = 800 with 800x500 lattice nodes and 100 immersed boundary markers running on Nvidia Geforce 1080 (plotting every 50 time steps).


<p align="center">
    <img src ="assets/ldc.gif"/>
</p>

> Simulation of lid-driven cavity at Re = 20000 with 1000x1000 lattice nodes (takes about 5 min on Nvidia Geforce 1080).

## Getting Started

To locally install for development:

```bash
git clone https://github.com/haimingz/vivsim.git
cd vivsim
pip install -r requirements.txt
pip install -e .
```