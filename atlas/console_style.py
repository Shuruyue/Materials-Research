"""
Console style helpers for consistent colored terminal output.

This module only affects display style and does not change message content.
"""

from __future__ import annotations

import builtins
import os
import re
import sys
from typing import Any

_ANSI_RESET = "\x1b[0m"
_ANSI_BOLD = "\x1b[1m"
_ANSI_DIM = "\x1b[2m"
_ANSI_RED = "\x1b[31m"
_ANSI_GREEN = "\x1b[32m"
_ANSI_YELLOW = "\x1b[33m"
_ANSI_BLUE = "\x1b[34m"
_ANSI_CYAN = "\x1b[36m"

_TOKEN_STYLES: list[tuple[str, str]] = [
    ("[ERROR]", _ANSI_BOLD + _ANSI_RED),
    ("[WARN]", _ANSI_BOLD + _ANSI_YELLOW),
    ("[OK]", _ANSI_BOLD + _ANSI_GREEN),
    ("[INFO]", _ANSI_CYAN),
    ("[RUN]", _ANSI_BLUE),
    ("[Mode]", _ANSI_BLUE),
]

_PHASE_HEADER_RE = re.compile(r"^\s*\[Phase\d+\]")


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_falsy(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"0", "false", "no", "off"}


def _supports_color(stream: Any) -> bool:
    if os.environ.get("NO_COLOR"):
        return False

    if _env_truthy("FORCE_COLOR") or _env_truthy("CLICOLOR_FORCE"):
        return True
    if _env_falsy("FORCE_COLOR") or _env_falsy("CLICOLOR"):
        return False
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False

    isatty = getattr(stream, "isatty", None)
    if callable(isatty):
        try:
            return bool(isatty())
        except (OSError, TypeError, ValueError):
            return False
    return False


def _style_line(line: str) -> str:
    raw = line.rstrip("\r\n")
    suffix = line[len(raw):]
    stripped = raw.strip()

    if not stripped:
        return line
    if "\x1b[" in raw:
        # Already styled text, keep as-is.
        return line

    if _PHASE_HEADER_RE.match(raw):
        return f"{_ANSI_BOLD}{_ANSI_BLUE}{raw}{_ANSI_RESET}{suffix}"

    if stripped and all(ch == "=" for ch in stripped):
        return f"{_ANSI_DIM}{raw}{_ANSI_RESET}{suffix}"
    if stripped and all(ch == "-" for ch in stripped):
        return f"{_ANSI_DIM}{raw}{_ANSI_RESET}{suffix}"

    for token, style in _TOKEN_STYLES:
        if token in raw:
            return f"{style}{raw}{_ANSI_RESET}{suffix}"

    return line


def _style_text(text: str) -> str:
    parts = text.splitlines(keepends=True)
    if not parts:
        return text
    return "".join(_style_line(part) for part in parts)


def install_console_style() -> None:
    """
    Install global print styling once per process.

    It preserves original text and only adds ANSI colors for terminal display.
    """
    if _env_falsy("ATLAS_CONSOLE_STYLE"):
        return
    if getattr(builtins.print, "_atlas_console_style_installed", False):
        return

    original_print = builtins.print

    def styled_print(*args: Any, **kwargs: Any) -> None:
        stream = kwargs.get("file")
        if stream is None:
            stream = sys.stdout
        if not _supports_color(stream):
            original_print(*args, **kwargs)
            return

        sep = kwargs.get("sep", " ")
        if sep is None:
            sep = " "
        text = sep.join(str(arg) for arg in args)
        styled = _style_text(text)

        new_kwargs = dict(kwargs)
        new_kwargs["sep"] = ""
        original_print(styled, **new_kwargs)

    styled_print._atlas_console_style_installed = True
    builtins.print = styled_print
