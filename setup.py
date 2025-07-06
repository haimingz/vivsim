from setuptools import setup, find_packages

setup(
    name="vivsim",
    version="1.0.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
        "matplotlib",
        "jax[cuda12]"
    ],
    python_requires=">=3.8",
)