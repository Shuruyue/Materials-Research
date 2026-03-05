"""Unit tests for console style helpers."""

from __future__ import annotations

import builtins
import io

from atlas.console_style import _style_line, _supports_color, install_console_style


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


class _NonTTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_supports_color_respects_no_color(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert _supports_color(_TTYBuffer()) is False


def test_supports_color_respects_force_and_term(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert _supports_color(_NonTTYBuffer()) is True
    monkeypatch.setenv("FORCE_COLOR", "0")
    monkeypatch.setenv("TERM", "dumb")
    assert _supports_color(_TTYBuffer()) is False


def test_install_console_style_handles_sep_none(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    original_print = builtins.print
    try:
        install_console_style()
        buf = _TTYBuffer()
        builtins.print("alpha", "beta", sep=None, file=buf, end="")
        output = buf.getvalue()
        assert "alpha beta" in output
    finally:
        builtins.print = original_print


def test_install_console_style_respects_disable_env(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_CONSOLE_STYLE", "0")
    original_print = builtins.print
    try:
        install_console_style()
        assert builtins.print is original_print
    finally:
        builtins.print = original_print


def test_style_line_matches_multi_digit_phase_header() -> None:
    styled = _style_line("[Phase10] benchmark summary")
    assert "\x1b[1m" in styled
    assert "\x1b[34m" in styled
