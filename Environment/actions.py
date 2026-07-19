"""
MegaBonk AI — Action Controller
=================================
Maps discrete action indices from the RL agent to keyboard keys
and mouse movements using PyDirectInput.

Action Space: MultiDiscrete([3, 3, 2, 9, 7])
  [0] Horizontal:  0=nothing, 1=A(left), 2=D(right)
  [1] Vertical:    0=nothing, 1=W(forward), 2=S(backward)
  [2] Jump:        0=nothing, 1=Space
  [3] Mouse ΔX:    0..8 → mapped to deltas [-4, -3, -2, -1, 0, +1, +2, +3, +4]
  [4] Mouse ΔY:    0..6 → mapped to deltas [-3, -2, -1, 0, +1, +2, +3]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pydirectinput

logger = logging.getLogger(__name__)

# Disable PyDirectInput's built-in pause between calls
pydirectinput.PAUSE = 0.0


@dataclass
class ActionConfig:
    """Configuration for the action controller."""

    # Mouse sensitivity: each delta unit = this many pixels of movement
    mouse_speed: int = 15
    # Smooth mouse over N sub-steps (1 = no smoothing)
    mouse_smoothing_steps: int = 1
    # How long to hold keys between steps (seconds)
    key_hold_duration: float = 0.05

    # Action space dimensions
    horizontal_options: int = 3
    vertical_options: int = 3
    jump_options: int = 2
    mouse_dx_options: int = 9
    mouse_dy_options: int = 7


# Maps action indices to key names
_HORIZONTAL_KEYS = {0: None, 1: "a", 2: "d"}
_VERTICAL_KEYS = {0: None, 1: "w", 2: "s"}
_JUMP_KEYS = {0: None, 1: "space"}


class ActionController:
    """
    Translates MultiDiscrete actions into game inputs.

    Manages key press/release state to avoid stuck keys,
    and applies mouse deltas with configurable sensitivity.

    Usage:
        controller = ActionController(config)
        controller.execute(action_array)  # np.array([1, 2, 0, 5, 3])
        controller.release_all()  # cleanup
    """

    def __init__(self, config: Optional[ActionConfig] = None) -> None:
        self.config = config or ActionConfig()

        # Track currently pressed keys so we can release them
        self._pressed_keys: set[str] = set()

        # Pre-compute mouse delta lookup tables
        # ΔX: 9 options → [-4, -3, -2, -1, 0, +1, +2, +3, +4]
        n_dx = self.config.mouse_dx_options
        self._dx_table = np.arange(n_dx) - (n_dx // 2)

        # ΔY: 7 options → [-3, -2, -1, 0, +1, +2, +3]
        n_dy = self.config.mouse_dy_options
        self._dy_table = np.arange(n_dy) - (n_dy // 2)

        logger.info(
            "ActionController initialized: mouse_speed=%d, dx_range=%s, dy_range=%s",
            self.config.mouse_speed,
            self._dx_table.tolist(),
            self._dy_table.tolist(),
        )

    def execute(self, action: np.ndarray) -> dict:
        """
        Execute a MultiDiscrete action.

        Args:
            action: numpy array of shape (5,) with values:
                [horizontal, vertical, jump, mouse_dx, mouse_dy]

        Returns:
            Dict with action details for logging/debugging.
        """
        horizontal = int(action[0])
        vertical = int(action[1])
        jump = int(action[2])
        mouse_dx_idx = int(action[3])
        mouse_dy_idx = int(action[4])

        # --- Keyboard ---
        target_keys: set[str] = set()

        h_key = _HORIZONTAL_KEYS.get(horizontal)
        if h_key:
            target_keys.add(h_key)

        v_key = _VERTICAL_KEYS.get(vertical)
        if v_key:
            target_keys.add(v_key)

        j_key = _JUMP_KEYS.get(jump)
        if j_key:
            target_keys.add(j_key)

        # Release keys that should no longer be held
        keys_to_release = self._pressed_keys - target_keys
        for key in keys_to_release:
            pydirectinput.keyUp(key)

        # Press keys that aren't already held
        keys_to_press = target_keys - self._pressed_keys
        for key in keys_to_press:
            pydirectinput.keyDown(key)

        self._pressed_keys = target_keys

        # --- Mouse ---
        raw_dx = int(self._dx_table[mouse_dx_idx])
        raw_dy = int(self._dy_table[mouse_dy_idx])

        pixel_dx = raw_dx * self.config.mouse_speed
        pixel_dy = raw_dy * self.config.mouse_speed

        if pixel_dx != 0 or pixel_dy != 0:
            self._move_mouse(pixel_dx, pixel_dy)

        return {
            "keys": list(target_keys),
            "mouse_dx": raw_dx,
            "mouse_dy": raw_dy,
            "pixel_dx": pixel_dx,
            "pixel_dy": pixel_dy,
        }

    def _move_mouse(self, pixel_dx: int, pixel_dy: int) -> None:
        """
        Move the mouse with optional smoothing.

        Args:
            pixel_dx: Total horizontal pixel movement.
            pixel_dy: Total vertical pixel movement.
        """
        steps = self.config.mouse_smoothing_steps

        if steps <= 1:
            # Single movement — fastest
            pydirectinput.moveRel(pixel_dx, pixel_dy, relative=True)
        else:
            # Smooth over N sub-steps
            step_dx = pixel_dx / steps
            step_dy = pixel_dy / steps
            for i in range(steps):
                # Round carefully to avoid drift
                dx = int(round(step_dx * (i + 1))) - int(round(step_dx * i))
                dy = int(round(step_dy * (i + 1))) - int(round(step_dy * i))
                pydirectinput.moveRel(dx, dy, relative=True)
                if i < steps - 1:
                    time.sleep(0.001)

    def release_all(self) -> None:
        """Release all currently pressed keys. Call this on episode end / cleanup."""
        for key in self._pressed_keys:
            try:
                pydirectinput.keyUp(key)
            except Exception:
                pass
        self._pressed_keys.clear()
        logger.debug("All keys released")

    def click(self, x: int, y: int, button: str = "left") -> None:
        """
        Click at an absolute screen position. Useful for menu navigation.

        Args:
            x: Screen X coordinate.
            y: Screen Y coordinate.
            button: "left" or "right".
        """
        pydirectinput.click(x, y, button=button)

    def press_key(self, key: str, duration: float = 0.1) -> None:
        """
        Press and release a single key. Useful for menu navigation.

        Args:
            key: Key name (e.g., "enter", "escape").
            duration: How long to hold the key.
        """
        pydirectinput.keyDown(key)
        time.sleep(duration)
        pydirectinput.keyUp(key)

    @property
    def action_space_dims(self) -> list[int]:
        """Return the MultiDiscrete dimensions for gymnasium."""
        return [
            self.config.horizontal_options,
            self.config.vertical_options,
            self.config.jump_options,
            self.config.mouse_dx_options,
            self.config.mouse_dy_options,
        ]

    def __del__(self) -> None:
        self.release_all()
