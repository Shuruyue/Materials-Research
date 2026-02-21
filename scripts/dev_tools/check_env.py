"""
ATLAS environment checker.

Usage:
    python scripts/dev_tools/check_env.py
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass


@dataclass(frozen=True)
class DependencyCheck:
    package: str
    import_name: str
    required: bool
    group: str
    hint: str


CHECKS = [
    DependencyCheck("torch", "torch", True, "core", "pip install -r requirements.txt"),
    DependencyCheck("torch-geometric", "torch_geometric", True, "core", "pip install -r requirements.txt"),
    DependencyCheck("numpy", "numpy", True, "core", "pip install -r requirements.txt"),
    DependencyCheck("pandas", "pandas", True, "core", "pip install -r requirements.txt"),
    DependencyCheck("pymatgen", "pymatgen", True, "core", "pip install -r requirements.txt"),
    DependencyCheck("ase", "ase", True, "core", "pip install -r requirements.txt"),
    DependencyCheck("e3nn", "e3nn", True, "core", "pip install -r requirements.txt"),
    DependencyCheck("jarvis-tools", "jarvis", True, "core", "pip install -r requirements.txt"),
    DependencyCheck("rustworkx", "rustworkx", True, "core", "pip install -r requirements.txt"),
    DependencyCheck("botorch", "botorch", True, "core", "pip install -r requirements.txt"),
    DependencyCheck("gpytorch", "gpytorch", True, "core", "pip install -r requirements.txt"),
    DependencyCheck("torch-scatter", "torch_scatter", False, "optional", "install wheel matching torch/pyg"),
    DependencyCheck("torch-sparse", "torch_sparse", False, "optional", "install wheel matching torch/pyg"),
    DependencyCheck("mace-torch", "mace", False, "mace", "pip install -e .[mace]"),
    DependencyCheck("matbench", "matbench", False, "benchmark", "pip install -e .[benchmark]"),
    DependencyCheck("matminer", "matminer", False, "benchmark", "pip install -e .[benchmark]"),
]


def _module_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def run_checks() -> int:
    print("=" * 72)
    print("ATLAS Environment Check")
    print("=" * 72)

    missing_required = []
    missing_optional = []

    for check in CHECKS:
        try:
            importlib.import_module(check.import_name)
            version = _module_version(check.package)
            print(f"[OK]    {check.package:<18} version={version} ({check.group})")
        except Exception as exc:
            msg = f"[MISSING] {check.package:<18} ({check.group}) -> {check.hint}"
            if check.required:
                missing_required.append((check.package, str(exc)))
                print(msg)
            else:
                missing_optional.append((check.package, str(exc)))
                print(msg)

    # CUDA / device info
    try:
        import torch

        print("-" * 72)
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        print(f"PyTorch CUDA version  : {torch.version.cuda}")
        print(f"GPU count             : {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"GPU[0] name           : {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"CUDA inspection skipped: {exc}")

    print("-" * 72)
    if missing_required:
        print("Required dependencies missing:")
        for pkg, err in missing_required:
            print(f"  - {pkg}: {err}")
        print("Suggested command: pip install -r requirements.txt && pip install -e .")
        return 1

    if missing_optional:
        print("Optional dependencies missing (allowed):")
        for pkg, err in missing_optional:
            print(f"  - {pkg}: {err}")
        print("Install optional profiles only if needed for your phase.")
    else:
        print("All required and optional dependencies detected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_checks())
