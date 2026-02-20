# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""
Model test classes composed from hierarchical test mixins.

The mixins are organized hierarchically:
    BasicModelTestsMixin (base fixtures + basic tests)
    └─ EnergyModelTestsMixin (adds energy-specific tests)
       ├─ CompilationTestsMixin (adds compilation tests for energy models)
       └─ LAMMPSMLIAPIntegrationMixin (adds LAMMPS ML-IAP integration tests)

This module provides standard test class compositions:
- BaseEnergyModelTests: Everything (basic + energy + compilation + LAMMPS ML-IAP)

For fine-grained control, compose from mixins directly:
    from nequip.utils.unittests.model_tests_compilation import CompilationTestsMixin
    from nequip.utils.unittests.model_tests_lammps import LAMMPSMLIAPIntegrationMixin

    class MyModelTests(CompilationTestsMixin, LAMMPSMLIAPIntegrationMixin):
        # Gets basic + energy + compilation + LAMMPS ML-IAP via inheritance
        ...
"""

from .model_tests_compilation import CompilationTestsMixin
from .model_tests_lammps import LAMMPSMLIAPIntegrationMixin


# see https://github.com/pytest-dev/pytest/issues/421#issuecomment-943386533
# to allow external packages to import tests through subclassing
class BaseEnergyModelTests(CompilationTestsMixin, LAMMPSMLIAPIntegrationMixin):
    """
    Standard energy model tests: includes all test types.

    This class composes:
    - CompilationTestsMixin → EnergyModelTestsMixin → BasicModelTestsMixin
    - LAMMPSMLIAPIntegrationMixin → EnergyModelTestsMixin → BasicModelTestsMixin

    Via the inheritance hierarchy, this includes all tests:
    - Basic tests (init, forward, equivariance, batching, etc.)
    - Energy-specific tests (large separation, gradients, forces, smoothness, etc.)
    - Compilation tests (nequip-compile, train-time compile)
    - LAMMPS ML-IAP integration tests (not C++ pair style)

    Subclasses must provide:
    - `config` fixture: model configuration dict
    - `strict_locality` fixture: True if strictly local, False if message-passing

    For models without LAMMPS ML-IAP support:
        from nequip.utils.unittests.model_tests_compilation import CompilationTestsMixin

        class MyModelTests(CompilationTestsMixin):
            # basic + energy + compilation, skip LAMMPS ML-IAP
            ...
    """

    pass
