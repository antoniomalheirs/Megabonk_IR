"""
Windows helpers for finding and focusing the MegaBonk game window.

The functions in this module use only the Python standard library so users do
not need to install pywin32 just to calibrate a capture region.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import platform


@dataclass(frozen=True)
class WindowInfo:
    """Basic information for a visible top-level Windows window."""

    hwnd: int
    title: str
    rect: tuple[int, int, int, int]


class WindowLookupError(RuntimeError):
    """Raised when a requested game window cannot be found."""


def _require_windows() -> None:
    if platform.system() != "Windows":
        raise WindowLookupError("Window lookup is only available on Windows.")


def _user32():
    _require_windows()
    return ctypes.windll.user32


def _get_window_text(hwnd: int) -> str:
    user32 = _user32()
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def _get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    user32 = _user32()
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise WindowLookupError(f"Could not read window rect for hwnd={hwnd}.")
    return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


def list_windows() -> list[WindowInfo]:
    """Return visible top-level windows with non-empty titles."""
    user32 = _user32()
    windows: list[WindowInfo] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def enum_proc(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        title = _get_window_text(hwnd)
        if title:
            left, top, right, bottom = _get_window_rect(hwnd)
            if right > left and bottom > top:
                windows.append(WindowInfo(int(hwnd), title, (left, top, right, bottom)))
        return True

    user32.EnumWindows(enum_proc, 0)
    return windows


def find_window(title_contains: str) -> WindowInfo:
    """Find the first visible window whose title contains ``title_contains``."""
    needle = title_contains.casefold()
    matches = [window for window in list_windows() if needle in window.title.casefold()]
    if not matches:
        raise WindowLookupError(f"No visible window title contains: {title_contains!r}")
    return matches[0]


def focus_window(hwnd: int) -> None:
    """Bring a window to the foreground best-effort."""
    user32 = _user32()
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.SetForegroundWindow(hwnd)
