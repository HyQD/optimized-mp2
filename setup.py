from setuptools import setup, find_packages

setup(
    name="Optimized MP2",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "scipy",
        "opt_einsum",
        "quantum-systems @ git+https://github.com/HyQD/quantum-systems",
    ],
)
