"""Small helper for controlling ANSI terminal output."""

from __future__ import annotations

import os
import select
import shutil
import sys
import termios
import tty
from typing import List, Optional, Tuple

TermiosAttr = List[int | List[bytes | int]]


class TerminalController:
    """Context manager that prepares the terminal for smooth animations."""

    def __init__(self, *, clear: bool = True) -> None:
        self._clear = clear
        self._cursor_hidden = False
        self._stdin_fd: Optional[int] = None
        self._termios_before: Optional[TermiosAttr] = None
        self._input_enabled = False

    def __enter__(self) -> "TerminalController":
        if self._clear:
            sys.stdout.write("\033[2J")
        sys.stdout.write("\033[H")
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()
        self._cursor_hidden = True

        if sys.stdin.isatty():
            fd = sys.stdin.fileno()
            self._stdin_fd = fd
            try:
                self._termios_before = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                self._input_enabled = True
            except termios.error:
                self._termios_before = None
                self._stdin_fd = None
                self._input_enabled = False
        else:
            self._stdin_fd = None
            self._termios_before = None
            self._input_enabled = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()

    def restore(self) -> None:
        if self._cursor_hidden:
            sys.stdout.write("\033[0m")
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
            self._cursor_hidden = False

        if self._input_enabled and self._stdin_fd is not None and self._termios_before is not None:
            try:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._termios_before)
            except termios.error:
                pass
        self._input_enabled = False
        self._stdin_fd = None
        self._termios_before = None

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

    def poll_keys(self) -> List[str]:
        if not self._input_enabled or self._stdin_fd is None:
            return []

        keys: List[str] = []
        try:
            while True:
                readable, _, _ = select.select([sys.stdin], [], [], 0)
                if not readable:
                    break

                data = os.read(self._stdin_fd, 1)
                if not data:
                    break

                char = data.decode("utf-8", errors="ignore")
                if not char:
                    continue

                if char == "\x03":
                    raise KeyboardInterrupt

                if char == "\x1b":
                    sequence = self._read_escape_sequence()
                    key = self._map_escape_sequence(sequence)
                    if key is not None:
                        keys.append(key)
                    continue

                keys.append(char)
        except OSError:
            return keys

        return keys

    def _read_escape_sequence(self) -> str:
        sequence = "\x1b"
        if self._stdin_fd is None:
            return sequence

        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if not readable:
                break
            data = os.read(self._stdin_fd, 1)
            if not data:
                break
            char = data.decode("utf-8", errors="ignore")
            if not char:
                continue
            sequence += char
            if char.isalpha() or char == "~":
                break
        return sequence

    @staticmethod
    def _map_escape_sequence(sequence: str) -> Optional[str]:
        if not sequence:
            return None
        mapping = {
            "\x1b[A": "UP",
            "\x1b[B": "DOWN",
            "\x1b[C": "RIGHT",
            "\x1b[D": "LEFT",
        }
        if sequence in mapping:
            return mapping[sequence]
        if sequence.startswith("\x1b[") and sequence[-1] in "ABCD":
            return mapping.get("\x1b[" + sequence[-1], None)
        return None
