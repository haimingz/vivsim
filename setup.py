from setuptools import setup, find_packages

setup(
    name="vivsim",
    version="0.5.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
        "matplotlib",
        "jax[cuda12]"
    ],
    python_requires=">=3.8",
)