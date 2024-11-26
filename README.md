<p align="center">
<img src ="assets/vivsim.svg"/>
</p>

[![GitHub Clones](https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count&url=https://gist.githubusercontent.com/haimingz/8981033dc17c32a7c2f409e631b57309/raw/clone.json&logo=github)](https://github.com/MShawon/github-clone-count-badge)

## VIVSIM

VIVSIM is a Python library for accelerated fluid-structure interaction (FSI) simulations based on the immersed boundary -lattice Boltzmann method (IB-LBM). It was originated from a research project requiring efficient simulation codes for studying vortex-induced vibration (VIV) of underwater structures. 

Similar to projects like [JAX-CFD](https://github.com/google/jax-cfd) and [XLB](https://github.com/Autodesk/XLB), VIVSIM utilizes [JAX](https://github.com/jax-ml/jax) as the backend to harness the power of hardware accelerators (mainly GPUs).

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
- Single Relaxation Time (SRT) model
- Multiple Relaxation Time (MRT) model

Boundary Conditions:
- Velocity boundary using Non-Equilibrium Bounce Back (NEBB) method
- No-slip boundary using Halfway Bounce-Back method
- Outflow boundary simply by copying the second last column/row
- Periodic boundary

Fluid-Structure Interaction
- Multi Direct-Forcing Immersed Boundary method.
- Confined Immersed Boundary technique to speed up VIV simulation

## Getting Started

To locally install VIVSIM for development:

```bash
git clone https://github.com/haimingz/vivsim.git
pip install -e vivsim
```
This package is based on JAX, whose installation may depend on the OS and hardware. If the above command does not work well, please refer to the [JAX Documentation](https://jax.readthedocs.io/en/latest/installation.html) for the latest installation guidance. 

Alternatively, you can run the following command in a cell on Google Colab to install VIVSIM and run simulations using free/paid GPU on the cloud.

```python
!pip install git+https://github.com/haimingz/vivsim
```
You can also create a Singularity image from the provided definition file `vivsim.def` and execute your code on High Performance Computing (HPC) clusters. 

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

