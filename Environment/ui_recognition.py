"""
MegaBonk AI — UI Recognition
=============================
Detects game-interface state that should be exposed to the RL policy.

The recognizer is intentionally configurable because MegaBonk UI positions and
prompts can vary by resolution, language, scaling and patch version. Calibrate
or add templates under ``Configs/templates`` and wire them in ``ui`` config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UIRecognitionConfig:
    """Configuration for high-level UI detection."""

    enabled: bool = True
    template_match_threshold: float = 0.8
    templates: dict[str, str] = field(default_factory=dict)
    regions: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    score_normalizer: float = 10000.0


class UIRecognizer:
    """Recognizes menus, perk-choice screens, prompts and HUD-derived state."""

    def __init__(
        self,
        config: Optional[UIRecognitionConfig] = None,
        project_root: str = ".",
    ) -> None:
        self.config = config or UIRecognitionConfig()
        self._project_root = Path(project_root)
        self._templates: dict[str, np.ndarray] = {}

        for name, path in self.config.templates.items():
            self._load_template(name, path)

    def recognize(self, frame: np.ndarray, reward_info: Optional[dict] = None) -> dict:
        """
        Return machine-friendly UI features for the current frame.

        Values are numeric/bool so they can be converted into semantic planes for
        the neural network, logged in ``info`` and used by automation helpers.
        """
        reward_info = reward_info or {}
        features: dict = {
            "ui_enabled": self.config.enabled,
            "hp": reward_info.get("hp"),
            "xp": reward_info.get("xp"),
            "score": reward_info.get("score"),
            "dead": bool(reward_info.get("dead", False)),
            "level_up_screen": bool(reward_info.get("level_up_screen", False)),
            "perk_choice": bool(reward_info.get("level_up_screen", False)),
            "interactable_prompt": False,
            "choice_screen": False,
            "blocking_menu": False,
            "run_summary": False,
            "main_menu": False,
            "pause_menu": False,
            "stage_select": False,
            "loading_screen": False,
        }

        if not self.config.enabled:
            return features

        threshold = self.config.template_match_threshold
        for name in self._templates:
            detected = self._detect_template(frame, name, threshold)
            features[name] = detected

            # Common aliases used by the environment/model.
            if name in {"perk_choice", "level_up", "level_up_screen", "upgrade_choice", "item_choice", "weapon_choice", "tome_choice", "chest_reward"}:
                features["perk_choice"] = features["perk_choice"] or detected
                features["level_up_screen"] = features["level_up_screen"] or detected
                features["choice_screen"] = features["choice_screen"] or detected
            elif name in {"interactable", "interact_prompt", "pickup_prompt", "shrine_prompt", "chest_prompt", "npc_prompt", "portal_prompt", "challenge_prompt"}:
                features["interactable_prompt"] = (
                    features["interactable_prompt"] or detected
                )
            elif name in {"main_menu", "stage_select", "character_select", "shop", "quests", "unlocks", "settings", "credits", "confirmation_dialog", "pause_menu", "loading_screen"}:
                features["blocking_menu"] = features["blocking_menu"] or detected
            elif name in {"game_over", "death_screen", "results_screen", "run_summary"}:
                features["run_summary"] = features["run_summary"] or detected
            elif name in features:
                features[name] = detected

        features["blocking_menu"] = features["blocking_menu"] or any(
            bool(features.get(name, False))
            for name in (
                "main_menu",
                "stage_select",
                "character_select",
                "shop",
                "quests",
                "unlocks",
                "settings",
                "credits",
                "confirmation_dialog",
                "pause_menu",
                "loading_screen",
            )
        )

        return features

    def _load_template(self, name: str, path: str) -> bool:
        full_path = self._project_root / path
        if not full_path.exists():
            logger.warning("UI template not found: %s", full_path)
            return False

        template = cv2.imread(str(full_path))
        if template is None:
            logger.warning("Failed to read UI template: %s", full_path)
            return False

        self._templates[name] = template
        logger.info("Loaded UI template '%s' from %s", name, full_path)
        return True

    def _detect_template(self, frame: np.ndarray, name: str, threshold: float) -> bool:
        template = self._templates.get(name)
        if template is None:
            return False

        search_frame = frame
        region = self.config.regions.get(name)
        if region is not None:
            left, top, right, bottom = region
            search_frame = frame[top:bottom, left:right]
            if search_frame.size == 0:
                return False

        if search_frame.shape[0] < template.shape[0] or search_frame.shape[1] < template.shape[1]:
            return False

        result = cv2.matchTemplate(search_frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return bool(max_val >= threshold)
