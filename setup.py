import setuptools
  
setuptools.setup(
   name='vivsim',
   packages=setuptools.find_packages(),
   version='1.0',
   description='Accelerated vortex-induced vibration (VIV) simulation using immersed boundary lattice Boltzmann method (IB-LBM) powered by JAX.',
   author='Haiming Zhu',
   author_email='zhuhaiming@gmail.com',
   url="https://github.com/haimingz/vivsim", 
   # install_requires=['numpy', 'jax', 'tqdm', 'matplotlib', 'jupyter'],
)
