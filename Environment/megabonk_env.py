"""
MegaBonk AI — Gymnasium Environment
======================================
Custom Gymnasium environment that wraps MegaBonk via screen capture,
OCR-based rewards, and keyboard/mouse input simulation.

Observation: Stacked grayscale frames (4, 84, 84)
Actions:     MultiDiscrete([3, 3, 2, 9, 7, 2]) — WASD, Jump, Mouse ΔX/ΔY, Interact
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from Capture.screen_capture import ScreenCapture
from Capture.preprocessing import FramePreprocessor
from Environment.actions import ActionController, ActionConfig
from Environment.rewards import RewardCalculator, RewardConfig
from Environment.ui_recognition import UIRecognitionConfig, UIRecognizer

logger = logging.getLogger(__name__)


def _key_sequence(value: Any, fallback: list[str] | None = None) -> list[str]:
    """Normalize a configured UI navigation key or key list."""
    if value is None:
        return list(fallback or [])
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(key) for key in value if key]
    return list(fallback or [])


class MegaBonkEnv(gym.Env):
    """
    Gymnasium environment for MegaBonk.

    Connects screen capture, frame preprocessing, action execution,
    and reward calculation into a standard RL interface.

    The environment captures the game screen, preprocesses it into
    stacked grayscale frames, and sends keyboard/mouse inputs based
    on the agent's actions. Rewards are computed from OCR readings
    of HUD elements (HP, XP, Score) and template-based detection
    of game events (death, level up).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        config: Optional[dict] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Initialize the MegaBonk environment.

        Args:
            config: Full configuration dict (from YAML).
                    If None, uses defaults.
            render_mode: "human" to display frames, None for headless.
        """
        super().__init__()
        self.render_mode = render_mode
        config = config or {}

        # --- Parse config sections ---
        cap_cfg = config.get("capture", {})
        pre_cfg = config.get("preprocessing", {})
        act_cfg = config.get("actions", {})
        rew_cfg = config.get("rewards", {})
        env_cfg = config.get("environment", {})
        ui_cfg = config.get("ui", {})

        # --- Screen Capture ---
        region = cap_cfg.get("region")
        self._capture = ScreenCapture(
            target_fps=cap_cfg.get("target_fps", 60),
            region=tuple(region) if region else None,
            output_color=cap_cfg.get("output_color", "BGR"),
        )

        # --- Preprocessing ---
        self._preprocessor = FramePreprocessor(
            width=pre_cfg.get("resize_width", 84),
            height=pre_cfg.get("resize_height", 84),
            grayscale=pre_cfg.get("grayscale", True),
            stack_size=pre_cfg.get("frame_stack", 4),
            semantic_ui_channels=pre_cfg.get("semantic_ui_channels", 3),
            score_normalizer=ui_cfg.get("score_normalizer", 10000.0),
        )

        # --- Action Controller ---
        self._action_config = ActionConfig(
            mouse_speed=act_cfg.get("mouse_speed", 15),
            mouse_smoothing_steps=act_cfg.get("mouse_smoothing_steps", 1),
            key_hold_duration=act_cfg.get("key_hold_duration", 0.05),
            horizontal_options=act_cfg.get("horizontal_options", 3),
            vertical_options=act_cfg.get("vertical_options", 3),
            jump_options=act_cfg.get("jump_options", 2),
            mouse_dx_options=act_cfg.get("mouse_dx_options", 9),
            mouse_dy_options=act_cfg.get("mouse_dy_options", 7),
            interact_options=act_cfg.get("interact_options", 2),
            interact_key=act_cfg.get("interact_key", "e"),
        )
        self._controller = ActionController(self._action_config)

        # --- Reward Calculator ---
        reward_config = RewardConfig(
            survival_reward=rew_cfg.get("survival_reward", 0.01),
            xp_reward=rew_cfg.get("xp_reward", 5.0),
            hp_loss_penalty=rew_cfg.get("hp_loss_penalty", 1.0),
            death_penalty=rew_cfg.get("death_penalty", -10.0),
            score_reward_multiplier=rew_cfg.get("score_reward_multiplier", 0.1),
            exploration_reward=rew_cfg.get("exploration_reward", 0.002),
            visual_novelty_threshold=rew_cfg.get("visual_novelty_threshold", 6.0),
            stagnation_penalty=rew_cfg.get("stagnation_penalty", 0.01),
            stagnation_steps=rew_cfg.get("stagnation_steps", 90),
            hp_region=tuple(rew_cfg.get("hp_region", [50, 50, 300, 80])),
            xp_region=tuple(rew_cfg.get("xp_region", [50, 85, 300, 105])),
            score_region=tuple(rew_cfg.get("score_region", [850, 20, 1070, 55])),
            game_over_template=rew_cfg.get("game_over_template"),
            level_up_template=rew_cfg.get("level_up_template"),
            template_match_threshold=rew_cfg.get("template_match_threshold", 0.8),
        )
        # Determine project root (parent of Environment/)
        project_root = str(Path(__file__).resolve().parent.parent)
        self._reward_calc = RewardCalculator(reward_config, project_root)

        # --- UI recognizer ---
        ui_regions = {
            name: tuple(region) for name, region in ui_cfg.get("regions", {}).items()
        }
        ui_config = UIRecognitionConfig(
            enabled=ui_cfg.get("enabled", True),
            template_match_threshold=ui_cfg.get(
                "template_match_threshold",
                rew_cfg.get("template_match_threshold", 0.8),
            ),
            templates=ui_cfg.get("templates", {}),
            regions=ui_regions,
            score_normalizer=ui_cfg.get("score_normalizer", 10000.0),
        )
        self._ui_recognizer = UIRecognizer(ui_config, project_root)

        # --- Environment settings ---
        self._max_steps = env_cfg.get("max_steps", 0)
        self._step_delay = env_cfg.get("step_delay", 0.033)
        self._reset_delay = env_cfg.get("reset_delay", 3.0)
        self._auto_confirm_level_up = env_cfg.get("auto_confirm_level_up", True)
        self._auto_confirm_key = env_cfg.get("auto_confirm_key", "enter")
        self._auto_confirm_cooldown_steps = env_cfg.get("auto_confirm_cooldown_steps", 30)
        self._auto_confirm_repeats = env_cfg.get("auto_confirm_repeats", 2)
        self._pending_auto_confirms = 0
        self._last_auto_confirm_step = -self._auto_confirm_cooldown_steps
        self._auto_interact_enabled = env_cfg.get("auto_interact_enabled", True)
        self._auto_interact_key = env_cfg.get(
            "auto_interact_key", self._action_config.interact_key
        )
        self._auto_interact_every_steps = env_cfg.get("auto_interact_every_steps", 45)
        self._auto_interact_hold_duration = env_cfg.get("auto_interact_hold_duration", 0.05)
        self._last_auto_interact_step = 0
        self._auto_navigate_ui = env_cfg.get("auto_navigate_ui", True)
        self._auto_menu_confirm_key = env_cfg.get("auto_menu_confirm_key", "enter")
        self._auto_death_restart_key = env_cfg.get("auto_death_restart_key", "enter")
        self._auto_pause_back_key = env_cfg.get("auto_pause_back_key", "escape")
        self._auto_ui_cooldown_steps = env_cfg.get("auto_ui_cooldown_steps", 20)
        self._ui_action_pause_seconds = env_cfg.get("ui_action_pause_seconds", 0.15)
        self._ui_release_before_input = env_cfg.get("ui_release_before_input", True)
        self._pause_agent_on_blocking_ui = env_cfg.get("pause_agent_on_blocking_ui", True)
        self._last_auto_ui_step = -self._auto_ui_cooldown_steps
        self._auto_ui_sequences = {
            "death_restart": _key_sequence(
                env_cfg.get("auto_death_restart_keys"),
                [self._auto_death_restart_key],
            ),
            "main_menu": _key_sequence(
                env_cfg.get("auto_main_menu_keys"),
                [self._auto_menu_confirm_key],
            ),
            "character_select": _key_sequence(
                env_cfg.get("auto_character_select_keys"),
                [self._auto_menu_confirm_key],
            ),
            "stage_select": _key_sequence(
                env_cfg.get("auto_stage_select_keys"),
                [self._auto_menu_confirm_key],
            ),
            "difficulty_select": _key_sequence(
                env_cfg.get("auto_difficulty_select_keys"),
                [self._auto_menu_confirm_key],
            ),
            "confirmation_dialog": _key_sequence(
                env_cfg.get("auto_confirmation_dialog_keys"),
                [self._auto_menu_confirm_key],
            ),
            "pause_back": _key_sequence(
                env_cfg.get("auto_pause_back_keys"),
                [self._auto_pause_back_key],
            ),
            "blocking_menu": _key_sequence(
                env_cfg.get("auto_blocking_menu_keys"),
                [self._auto_menu_confirm_key],
            ),
            "loading_screen": _key_sequence(
                env_cfg.get("auto_loading_screen_keys"),
                [],
            ),
        }
        self._auto_ui_sequence_positions = {
            name: 0 for name in self._auto_ui_sequences
        }

        # --- Spaces ---
        obs_shape = self._preprocessor.observation_shape  # (4, 84, 84)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8,
        )

        self.action_space = spaces.MultiDiscrete(
            self._controller.action_space_dims
        )

        # --- Internal state ---
        self._current_step = 0
        self._episode_reward = 0.0
        self._capture_started = False
        self._last_raw_frame: Optional[np.ndarray] = None

        logger.info(
            "MegaBonkEnv initialized: obs=%s, actions=%s",
            obs_shape,
            self._controller.action_space_dims,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment for a new episode.

        This releases all keys, waits for the game to be ready,
        and returns the initial observation.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed, options=options)

        # Release all keys from previous episode
        self._controller.release_all()

        # Start capture if not already running
        if not self._capture_started:
            self._capture.start()
            self._capture_started = True
            time.sleep(1.0)  # wait for first frames

        # Reset internal state
        self._current_step = 0
        self._episode_reward = 0.0
        self._preprocessor.reset()
        self._reward_calc.reset()
        self._pending_auto_confirms = 0
        self._last_auto_confirm_step = -self._auto_confirm_cooldown_steps
        self._last_auto_interact_step = 0
        self._last_auto_ui_step = -self._auto_ui_cooldown_steps
        self._auto_ui_sequence_positions = {
            name: 0 for name in self._auto_ui_sequences
        }

        # Wait for game to be ready (post-death screen, loading, etc.)
        time.sleep(self._reset_delay)
        self._last_auto_ui_step = -self._auto_ui_cooldown_steps

        # Capture initial observation
        frame = self._capture.get_latest_frame()
        if frame is None:
            # Fallback: create a black frame
            logger.warning("No frame available on reset, using black frame")
            h = self._preprocessor.height
            w = self._preprocessor.width
            frame = np.zeros((h, w, 3), dtype=np.uint8)

        self._last_raw_frame = frame
        initial_ui = self._ui_recognizer.recognize(frame, {})
        auto_ui = self._maybe_auto_navigate_ui(initial_ui, allow_immediate=True)
        observation = self._preprocessor.process(frame, initial_ui)

        info = {
            "episode_step": 0,
            "raw_frame_shape": frame.shape,
            "ui": initial_ui,
            "auto_ui": auto_ui,
        }
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: MultiDiscrete action array of shape (6,).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self._current_step += 1

        # Capture and classify the current UI before sending policy inputs.
        # Blocking menus/choice screens should receive deliberate UI navigation
        # keys instead of stale movement/mouse actions from the agent.
        frame = self._capture.get_latest_frame()
        if frame is None:
            frame = self._last_raw_frame
        pre_ui_info = None
        pre_auto_ui_info = {"triggered": False, "key": None, "reason": None}
        pre_auto_choice_info = {
            "triggered": False,
            "key": None,
            "pending": self._pending_auto_confirms,
        }
        if frame is not None:
            pre_reward_info = self._reward_calc.peek(frame)
            pre_ui_info = self._ui_recognizer.recognize(frame, pre_reward_info)
            if self._pause_agent_for_ui(pre_ui_info):
                self._controller.release_all()
                pre_auto_ui_info = self._maybe_auto_navigate_ui(pre_ui_info)
                pre_auto_choice_info = self._maybe_auto_confirm_choice(pre_ui_info)
                action_info = {
                    "keys": [],
                    "mouse_dx": 0,
                    "mouse_dy": 0,
                    "pixel_dx": 0,
                    "pixel_dy": 0,
                    "interact": False,
                    "suppressed_for_ui": True,
                }
                if (
                    not pre_auto_ui_info.get("triggered")
                    and not pre_auto_choice_info.get("triggered")
                    and self._ui_action_pause_seconds > 0
                ):
                    time.sleep(self._ui_action_pause_seconds)
            else:
                action_info = self._controller.execute(action)
        else:
            action_info = self._controller.execute(action)

        # Wait for the action/UI interaction to take effect
        if self._step_delay > 0:
            time.sleep(self._step_delay)

        # Capture the resulting frame
        frame = self._capture.get_latest_frame()
        if frame is None:
            frame = self._last_raw_frame
        else:
            self._last_raw_frame = frame
        if frame is None:
            logger.warning("No frame available on step, using black frame")
            frame = np.zeros(
                (self._preprocessor.height, self._preprocessor.width, 3),
                dtype=np.uint8,
            )
            self._last_raw_frame = frame

        # Calculate reward from the raw (full-res) frame
        reward, reward_info = self._reward_calc.calculate(frame)
        self._episode_reward += reward

        ui_info = self._ui_recognizer.recognize(frame, reward_info)
        auto_ui_info = pre_auto_ui_info
        auto_choice_info = pre_auto_choice_info
        if not auto_ui_info.get("triggered"):
            auto_ui_info = self._maybe_auto_navigate_ui(ui_info)
        if not auto_choice_info.get("triggered"):
            auto_choice_info = self._maybe_auto_confirm_choice(ui_info)
        auto_interact_info = self._maybe_auto_interact(action_info, ui_info)

        # Preprocess for the agent
        observation = self._preprocessor.process(frame, ui_info)

        # Check termination conditions
        terminated = reward_info.get("dead", False)
        truncated = (
            self._max_steps > 0 and self._current_step >= self._max_steps
        )

        # Build info dict
        info = {
            "episode_step": self._current_step,
            "episode_reward": self._episode_reward,
            "action": action_info,
            "pre_ui": pre_ui_info,
            "auto_choice": auto_choice_info,
            "auto_interact": auto_interact_info,
            "auto_ui": auto_ui_info,
            "ui": ui_info,
            **reward_info,
        }

        if terminated or truncated:
            # Release all keys on episode end
            self._controller.release_all()
            logger.info(
                "Episode ended: steps=%d, reward=%.2f, dead=%s",
                self._current_step,
                self._episode_reward,
                terminated,
            )

        return observation, reward, terminated, truncated, info

    def _pause_agent_for_ui(self, ui_info: dict) -> bool:
        """Return True when UI should own input for this step."""
        if not self._pause_agent_on_blocking_ui:
            return False
        return self._auto_ui_reason(ui_info) is not None or bool(
            ui_info.get("level_up", False)
            or ui_info.get("level_up_screen", False)
            or ui_info.get("perk_choice", False)
            or ui_info.get("choice_screen", False)
        )

    def _press_ui_key(self, key: str, duration: float = 0.05) -> None:
        """Safely send a menu/choice key with optional release and settle pause."""
        if self._ui_release_before_input:
            self._controller.release_all()
        self._controller.press_key(key, duration=duration)
        if self._ui_action_pause_seconds > 0:
            time.sleep(self._ui_action_pause_seconds)

    def _maybe_auto_navigate_ui(
        self, ui_info: dict, allow_immediate: bool = False
    ) -> dict:
        """Navigate blocking menu/death screens detected by UI recognition."""
        info = {"triggered": False, "key": None, "reason": None}
        if not self._auto_navigate_ui:
            return info

        steps_since_last = self._current_step - self._last_auto_ui_step
        if not allow_immediate and steps_since_last < self._auto_ui_cooldown_steps:
            return info

        reason = self._auto_ui_reason(ui_info)
        if reason is None:
            return info

        keys = self._auto_ui_sequences.get(reason, [])
        if not keys:
            info.update({"reason": reason})
            return info

        key = self._next_auto_ui_key(reason, keys)
        self._press_ui_key(key, duration=0.05)
        self._last_auto_ui_step = self._current_step
        info.update({"triggered": True, "key": key, "reason": reason})
        return info

    def _auto_ui_reason(self, ui_info: dict) -> str | None:
        """Return the most specific configured UI navigation reason."""
        if bool(ui_info.get("dead", False)) or bool(
            ui_info.get("game_over", False)
        ) or bool(ui_info.get("run_summary", False)):
            return "death_restart"
        if bool(ui_info.get("pause_menu", False)):
            return "pause_back"

        for reason in (
            "main_menu",
            "character_select",
            "stage_select",
            "difficulty_select",
            "confirmation_dialog",
            "loading_screen",
        ):
            if bool(ui_info.get(reason, False)):
                return reason

        if bool(ui_info.get("blocking_menu", False)):
            return "blocking_menu"
        return None

    def _next_auto_ui_key(self, reason: str, keys: list[str]) -> str:
        """Cycle through the configured key sequence for a recognized UI state."""
        position = self._auto_ui_sequence_positions.get(reason, 0)
        key = keys[position % len(keys)]
        self._auto_ui_sequence_positions[reason] = position + 1
        return key

    def _maybe_auto_confirm_choice(self, ui_info: dict) -> dict:
        """Confirm perk/skill choice screens when recognized by UI perception."""
        info = {"triggered": False, "key": None, "pending": self._pending_auto_confirms}
        if not self._auto_confirm_level_up:
            return info

        choice_detected = ui_info.get("level_up", False) or ui_info.get(
            "level_up_screen", False
        ) or ui_info.get("perk_choice", False)
        if choice_detected and self._pending_auto_confirms <= 0:
            self._pending_auto_confirms = max(1, self._auto_confirm_repeats)

        if self._pending_auto_confirms <= 0:
            info["pending"] = 0
            return info

        steps_since_last = self._current_step - self._last_auto_confirm_step
        if steps_since_last < self._auto_confirm_cooldown_steps:
            info["pending"] = self._pending_auto_confirms
            return info

        self._press_ui_key(self._auto_confirm_key, duration=0.05)
        self._pending_auto_confirms -= 1
        self._last_auto_confirm_step = self._current_step
        info.update(
            {
                "triggered": True,
                "key": self._auto_confirm_key,
                "pending": self._pending_auto_confirms,
            }
        )
        return info

    def _maybe_auto_interact(self, action_info: dict, ui_info: Optional[dict] = None) -> dict:
        """Press interact for recognized prompts, plus periodic safety taps."""
        info = {"triggered": False, "key": None}
        if not self._auto_interact_enabled or action_info.get("interact", False):
            return info

        ui_info = ui_info or {}
        prompt_detected = bool(ui_info.get("interactable_prompt", False))
        if not prompt_detected:
            if self._auto_interact_every_steps <= 0:
                return info
            steps_since_last = self._current_step - self._last_auto_interact_step
            if steps_since_last < self._auto_interact_every_steps:
                return info

        self._controller.press_key(
            self._auto_interact_key,
            duration=self._auto_interact_hold_duration,
        )
        self._last_auto_interact_step = self._current_step
        info.update({"triggered": True, "key": self._auto_interact_key})
        return info

    def render(self) -> Optional[np.ndarray]:
        """Render the current frame (for human visualization)."""
        if self.render_mode == "human" and self._last_raw_frame is not None:
            display = cv2.resize(self._last_raw_frame, (640, 360))
            cv2.imshow("MegaBonk AI", display)
            cv2.waitKey(1)
            return display
        return self._last_raw_frame

    def close(self) -> None:
        """Clean up resources."""
        self._controller.release_all()
        self._capture.stop()
        self._capture_started = False

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

        logger.info("MegaBonkEnv closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
