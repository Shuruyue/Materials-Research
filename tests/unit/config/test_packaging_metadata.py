"""Tests for packaging metadata coherence (pyproject/setup/requirements-dev)."""

from __future__ import annotations

from pathlib import Path

import tomllib


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_pyproject_includes_dev_extra_with_core_tooling():
    pyproject_path = _project_root() / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    optional = payload["project"]["optional-dependencies"]

    assert "dev" in optional
    dev_extra = optional["dev"]
    assert any(dep.startswith("pytest>=") for dep in dev_extra)
    assert any(dep.startswith("pytest-cov>=") for dep in dev_extra)
    assert any(dep.startswith("ruff>=") for dep in dev_extra)
    assert any(dep.startswith("build>=") for dep in dev_extra)


def test_requirements_dev_points_to_dev_extra():
    req_path = _project_root() / "requirements-dev.txt"
    lines = [line.strip() for line in req_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert "-e .[dev]" in lines


def test_setup_py_is_main_guarded_shim():
    setup_path = _project_root() / "setup.py"
    text = setup_path.read_text(encoding="utf-8")
    assert "pyproject.toml" in text
    assert 'if __name__ == "__main__":' in text
    assert "setup()" in text
