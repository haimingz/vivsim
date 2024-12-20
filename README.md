<a href='https://github.com/MShawon/github-clone-count-badge'><img alt='GitHub Traffic' src='https://img.shields.io/badge/dynamic/json?color=success&label=Views&query=count&url=https://gist.githubusercontent.com/haimingz/b2e30dd706e57413c1c1f688de82ef6e/raw/traffic.json&logo=github'></a>
<a href='https://github.com/MShawon/github-clone-count-badge'><img alt='GitHub Clones' src='https://img.shields.io/badge/dynamic/json?color=success&label=Clone&query=count&url=https://gist.githubusercontent.com/haimingz/8981033dc17c32a7c2f409e631b57309/raw/clone.json&logo=github'></a>
<img align="right" src ="assets/TJU_logo.png" width=80px/>

<p align="center" style="margin-top: 80px;">
<img src ="assets/vivsim.svg"/>
</p>

## VIVSIM

VIVSIM is a Python library for accelerated fluid-structure interaction (FSI) simulations based on the immersed boundary -lattice Boltzmann method (IB-LBM). It was originated from a research project requiring efficient simulation codes for studying vortex-induced vibration (VIV) of underwater structures. 

Similar to projects like [JAX-CFD](https://github.com/google/jax-cfd) and [XLB](https://github.com/Autodesk/XLB), VIVSIM utilizes [JAX](https://github.com/jax-ml/jax) as the backend to harness the power of hardware accelerators, achieving massive parallelism on GPU/GPUs. 

**What's New**: Now we can run multi-GPU and multi-gird simulations (check out the example scripts) 🎉

## Examples

|  |  |
|---------|-------------|
| ![Cavity](assets/cavity.gif) | ![Text](assets/text.gif) |
| Lid-driven cavity at $Re = 2\times 10^4$ on a 1000x1000 lattice grid. | Flow passes through some texts on a 1000x1000 lattice grid. |
| <img src="assets/viv_100.gif" width=700/> | <img src="assets/viv_10000.gif" width=420/> |
| VIV of a cylinder with $U_r = 5$ and $Re = 100$. The simulation was conducted on a 2000x3000 lattice grid with 400 immersed boundary markers, taking about 8 min on a Nvidia A100.| VIV of a cylinder with $U_r = 5$ and $Re = 1\times 10^4$. The simulation was conducted on a 8000x4000 lattice grid with 1200 immersed boundary markers, taking about 30 min on 8 Nvidia A800. |

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
- Outflow boundary by assigning equilibrium distribution values
- Periodic boundary

Fluid-Structure Interaction
- Multi Direct-Forcing Immersed Boundary method.


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

