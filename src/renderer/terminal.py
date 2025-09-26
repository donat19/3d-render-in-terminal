"""Small helper for controlling ANSI terminal output."""

from __future__ import annotations

import os
import shutil
import sys
from typing import Tuple


class TerminalController:
    """Context manager that prepares the terminal for smooth animations."""

    def __init__(self, *, clear: bool = True) -> None:
        self._clear = clear
        self._cursor_hidden = False

    def __enter__(self) -> "TerminalController":
        if self._clear:
            sys.stdout.write("\033[2J")
        sys.stdout.write("\033[H")
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()
        self._cursor_hidden = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()

    def restore(self) -> None:
        if self._cursor_hidden:
            sys.stdout.write("\033[0m")
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
            self._cursor_hidden = False

    def draw(self, frame: str) -> None:
        sys.stdout.write("\033[H")
        sys.stdout.write(frame)
        sys.stdout.write("\033[0m")
        sys.stdout.flush()

    def get_size(self) -> os.terminal_size:
        return shutil.get_terminal_size(fallback=(100, 40))

    def size_tuple(self) -> Tuple[int, int]:
        size = self.get_size()
        return size.columns, size.lines
