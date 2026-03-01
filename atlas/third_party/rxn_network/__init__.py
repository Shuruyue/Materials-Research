"""The reaction-network package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("reaction-network")
except PackageNotFoundError:
    # Vendored source tree fallback.
    __version__ = "0+local"
