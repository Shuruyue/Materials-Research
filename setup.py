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
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "pymatgen>=2024.1.1",
        "ase>=3.22.0",
        "torch>=2.1.0",
        "torch-geometric>=2.4.0",
        "jarvis-tools>=2024.1.25",
        "tqdm>=4.65",
        "joblib>=1.3.0",
        "e3nn>=0.5.0",
        "botorch>=0.9.0",
        "gpytorch>=1.11.0",
    ],
    extras_require={
        "mace": [
            "mace-torch>=0.3.0",
        ],
        "test": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "mock>=5.1.0",
        ],
        "jupyter": [
            "jupyter>=1.0",
            "ipywidgets>=8.0",
        ],
    },
)
