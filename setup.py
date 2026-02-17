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
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
        "pymatgen>=2024.1.27",
        "ase>=3.23.0",
        "torch>=2.2.0",
        "torch-geometric>=2.5.0",
        "jarvis-tools>=2024.1.25",
        "tqdm>=4.65",
        "e3nn>=0.5.1",
        "botorch>=0.10.0",
        "gpytorch>=1.11.0",
    ],
    extras_require={
        "test": [  # Explicit test extras
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "mock>=5.1.0",
        ],
    },
)
