"""Tests for atlas.potentials.relaxers.mlip_arena_relaxer."""

from __future__ import annotations

import pytest

from atlas.potentials.relaxers.mlip_arena_relaxer import NativeMlipArenaRelaxer


def test_native_mlip_arena_relaxer_validates_init_inputs():
    with pytest.raises(ValueError, match="steps"):
        NativeMlipArenaRelaxer(steps=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="symmetry must be boolean-like"):
        NativeMlipArenaRelaxer(symmetry="maybe")

    relaxer = NativeMlipArenaRelaxer(symmetry="false")
    assert relaxer.symmetry is False
