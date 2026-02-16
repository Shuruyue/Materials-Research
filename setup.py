from setuptools import setup, find_packages

setup(
    name="atlas",
    version="0.1.0",
    description="ATLAS: Accelerated Topological Learning And Screening",
    author="ATLAS Team",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.11",
        "pandas>=2.0",
        "pymatgen>=2024.1.1",
        "ase>=3.22",
        "torch>=2.1",
        "jarvis-tools>=2024.1",
        "tqdm>=4.65",
    ],
    extras_require={
        "ml": [
            "torch-geometric>=2.4",
            "e3nn>=0.5",
            "mace-torch>=0.3.0",
            "botorch>=0.9",
            "gpytorch>=1.11",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
)
