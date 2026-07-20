"""
MegaBonk AI — Reward System
==============================
Extracts game state from screen regions using OCR and template matching.
Calculates composite rewards based on HP changes, XP gains, score, and survival.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Reward system configuration."""

    # Reward values
    survival_reward: float = 0.01
    xp_reward: float = 5.0
    hp_loss_penalty: float = 1.0
    death_penalty: float = -10.0
    score_reward_multiplier: float = 0.1
    exploration_reward: float = 0.002
    visual_novelty_threshold: float = 6.0
    stagnation_penalty: float = 0.01
    stagnation_steps: int = 90

    # HUD regions [left, top, right, bottom]
    hp_region: tuple[int, int, int, int] = (50, 50, 300, 80)
    xp_region: tuple[int, int, int, int] = (50, 85, 300, 105)
    score_region: tuple[int, int, int, int] = (850, 20, 1070, 55)

    # Template matching
    game_over_template: Optional[str] = None
    level_up_template: Optional[str] = None
    template_match_threshold: float = 0.8


class OCRReader:
    """
    Reads numeric values from screen regions using EasyOCR.

    Lazy-initializes the OCR reader on first use to avoid
    slow import times affecting startup.
    """

    def __init__(self) -> None:
        self._reader = None

    def _ensure_reader(self) -> None:
        """Lazy-load EasyOCR reader."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(
                ["en"],
                gpu=True,
                verbose=False,
            )
            logger.info("EasyOCR reader initialized (GPU)")

    def read_number(self, image: np.ndarray) -> Optional[float]:
        """
        Read a numeric value from a cropped image region.

        Args:
            image: Cropped BGR image of a HUD element.

        Returns:
            Extracted number, or None if OCR fails.
        """
        self._ensure_reader()

        try:
            # Preprocess for better OCR: grayscale → threshold → invert
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Increase contrast
            gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
            # Threshold to binary
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

            results = self._reader.readtext(binary, detail=0, allowlist="0123456789./,%")

            if not results:
                return None

            # Join all detected text and extract numbers
            text = "".join(results).strip()
            # Try to parse as a number (handle formats like "100/100", "1,234", "50%")
            return self._parse_number(text)

        except Exception as e:
            logger.debug("OCR failed: %s", e)
            return None

    @staticmethod
    def _parse_number(text: str) -> Optional[float]:
        """
        Parse various number formats from OCR text.

        Handles: "1234", "1,234", "100/200" (returns first number),
                 "50%", etc.
        """
        # Remove commas and spaces
        text = text.replace(",", "").replace(" ", "")

        # If it's a fraction like "100/200", take the first number
        if "/" in text:
            parts = text.split("/")
            text = parts[0]

        # Remove % sign
        text = text.replace("%", "")

        # Extract digits and decimal point
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            return float(match.group(1))

        return None


class BarReader:
    """
    Reads HP/XP bar values by analyzing the fill percentage
    of colored bars (alternative to OCR — more robust for bars).

    Works by detecting the proportion of pixels in the bar region
    that match the bar's color (green for HP, blue/yellow for XP).
    """

    @staticmethod
    def read_bar_percentage(
        image: np.ndarray,
        color_lower: np.ndarray,
        color_upper: np.ndarray,
    ) -> float:
        """
        Estimate bar fill percentage by color masking.

        Args:
            image: Cropped BGR image of the bar region.
            color_lower: Lower HSV bound for the bar color.
            color_upper: Upper HSV bound for the bar color.

        Returns:
            Fill percentage (0.0 to 1.0).
        """
        if image is None or image.size == 0:
            return 0.0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, color_lower, color_upper)

        # Count filled pixels across horizontal axis
        total_width = mask.shape[1]
        if total_width == 0:
            return 0.0

        # For each column, check if any pixel matches
        col_filled = np.any(mask > 0, axis=0)
        fill_count = np.sum(col_filled)

        return float(fill_count / total_width)

    @staticmethod
    def read_hp_bar(image: np.ndarray) -> float:
        """Read HP bar fill percentage (assumes green/red bar)."""
        # Green HP bar: HSV range for green
        lower_green = np.array([35, 80, 80])
        upper_green = np.array([85, 255, 255])
        return BarReader.read_bar_percentage(image, lower_green, upper_green)

    @staticmethod
    def read_xp_bar(image: np.ndarray) -> float:
        """Read XP bar fill percentage (assumes blue/yellow bar)."""
        # Try blue first
        lower_blue = np.array([90, 80, 80])
        upper_blue = np.array([130, 255, 255])
        pct = BarReader.read_bar_percentage(image, lower_blue, upper_blue)

        # If blue doesn't work, try yellow/gold
        if pct < 0.01:
            lower_yellow = np.array([15, 80, 80])
            upper_yellow = np.array([45, 255, 255])
            pct = BarReader.read_bar_percentage(image, lower_yellow, upper_yellow)

        return pct


class TemplateDetector:
    """
    Detects specific game states (game over, level up) via template matching.
    """

    def __init__(self, project_root: str = ".") -> None:
        self._templates: dict[str, np.ndarray] = {}
        self._project_root = Path(project_root)

    def load_template(self, name: str, path: str) -> bool:
        """
        Load a template image from disk.

        Args:
            name: Template identifier (e.g., "game_over").
            path: Path to the template image (relative to project root).

        Returns:
            True if loaded successfully.
        """
        full_path = self._project_root / path
        if not full_path.exists():
            logger.warning("Template not found: %s", full_path)
            return False

        template = cv2.imread(str(full_path))
        if template is None:
            logger.warning("Failed to read template: %s", full_path)
            return False

        self._templates[name] = template
        logger.info("Loaded template '%s' from %s", name, full_path)
        return True

    def detect(
        self,
        frame: np.ndarray,
        template_name: str,
        threshold: float = 0.8,
    ) -> bool:
        """
        Check if a template exists in the frame.

        Args:
            frame: Full screen frame (BGR).
            template_name: Name of a previously loaded template.
            threshold: Minimum match confidence (0.0 - 1.0).

        Returns:
            True if the template is found with confidence >= threshold.
        """
        if template_name not in self._templates:
            return False

        template = self._templates[template_name]
        if (
            frame is None
            or frame.size == 0
            or frame.shape[0] < template.shape[0]
            or frame.shape[1] < template.shape[1]
        ):
            return False
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        return max_val >= threshold


class RewardCalculator:
    """
    Composite reward calculator for MegaBonk.

    Tracks game state across steps and computes rewards based on:
    - Survival time
    - HP changes (penalty for damage)
    - XP/level up events
    - Score increases
    - Death detection

    Usage:
        calc = RewardCalculator(config)
        calc.reset()
        reward, info = calc.calculate(frame)
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        project_root: str = ".",
    ) -> None:
        self.config = config or RewardConfig()

        # OCR and bar readers
        self._ocr = OCRReader()
        self._bar_reader = BarReader()
        self._template_detector = TemplateDetector(project_root)

        # Load templates if configured
        if self.config.game_over_template:
            self._template_detector.load_template(
                "game_over", self.config.game_over_template
            )
        if self.config.level_up_template:
            self._template_detector.load_template(
                "level_up", self.config.level_up_template
            )

        # State tracking
        self._prev_hp: Optional[float] = None
        self._prev_xp: Optional[float] = None
        self._prev_score: Optional[float] = None
        self._prev_level_up_visible: bool = False
        self._prev_novelty_frame: Optional[np.ndarray] = None
        self._stagnation_count: int = 0
        self._steps: int = 0

    def reset(self) -> None:
        """Reset state tracking for a new episode."""
        self._prev_hp = None
        self._prev_xp = None
        self._prev_score = None
        self._prev_level_up_visible = False
        self._prev_novelty_frame = None
        self._stagnation_count = 0
        self._steps = 0

    def peek(self, frame: np.ndarray) -> dict:
        """Read UI-critical state without mutating reward history.

        This lightweight pre-step pass intentionally avoids OCR and reward
        shaping so it can gate policy inputs while menus/choice screens are
        visible without changing episode rewards or previous HUD values.
        """
        is_dead = self._template_detector.detect(
            frame, "game_over", self.config.template_match_threshold
        )
        level_up_screen = self._template_detector.detect(
            frame, "level_up", self.config.template_match_threshold
        )

        hp_crop = self._crop_region(frame, self.config.hp_region)
        xp_crop = self._crop_region(frame, self.config.xp_region)
        return {
            "step": self._steps,
            "dead": is_dead,
            "level_up": level_up_screen,
            "level_up_screen": level_up_screen,
            "hp": self._bar_reader.read_hp_bar(hp_crop),
            "xp": self._bar_reader.read_xp_bar(xp_crop),
            "score": self._prev_score,
        }

    def calculate(self, frame: np.ndarray) -> tuple[float, dict]:
        """
        Calculate reward from the current frame.

        Args:
            frame: Full screen frame (BGR), original resolution.

        Returns:
            Tuple of (reward, info_dict).
            info_dict contains extracted values and reward breakdown.
        """
        self._steps += 1
        reward = 0.0
        info: dict = {"step": self._steps}

        # --- Check for death (game over) ---
        is_dead = self._template_detector.detect(
            frame, "game_over", self.config.template_match_threshold
        )
        if is_dead:
            reward += self.config.death_penalty
            info["dead"] = True
            info["level_up"] = False
            info["level_up_screen"] = False
            info["reward_breakdown"] = {"death": self.config.death_penalty}
            return reward, info

        info["dead"] = False
        info["level_up"] = False
        info["level_up_screen"] = False

        # --- Survival reward ---
        reward += self.config.survival_reward
        breakdown = {"survival": self.config.survival_reward}

        # --- HP reading (bar-based) ---
        hp_crop = self._crop_region(frame, self.config.hp_region)
        current_hp = self._bar_reader.read_hp_bar(hp_crop)

        if self._prev_hp is not None and current_hp < self._prev_hp:
            hp_loss = self._prev_hp - current_hp
            hp_penalty = -self.config.hp_loss_penalty * hp_loss
            reward += hp_penalty
            breakdown["hp_loss"] = hp_penalty

        info["hp"] = current_hp
        self._prev_hp = current_hp

        # --- XP reading (bar-based) ---
        xp_crop = self._crop_region(frame, self.config.xp_region)
        current_xp = self._bar_reader.read_xp_bar(xp_crop)

        # Detect level up: XP bar resets or a perk-choice template appears.
        level_up_screen = self._template_detector.detect(
            frame, "level_up", self.config.template_match_threshold
        )
        level_up_event = False
        if self._prev_xp is not None and current_xp < self._prev_xp - 0.3:
            # XP bar reset → level up!
            level_up_event = True
        if level_up_screen and not self._prev_level_up_visible:
            # Template just appeared → perk/skill choice screen opened.
            level_up_event = True

        if level_up_event:
            reward += self.config.xp_reward
            breakdown["level_up"] = self.config.xp_reward
            info["level_up"] = True

        info["level_up_screen"] = level_up_screen
        info["xp"] = current_xp
        self._prev_xp = current_xp
        self._prev_level_up_visible = level_up_screen

        # --- Visual exploration / anti-stuck shaping ---
        novelty = self._calculate_visual_novelty(frame)
        info["visual_novelty"] = novelty
        if novelty >= self.config.visual_novelty_threshold:
            reward += self.config.exploration_reward
            breakdown["exploration"] = self.config.exploration_reward
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1
            if self._stagnation_count >= self.config.stagnation_steps:
                reward -= self.config.stagnation_penalty
                breakdown["stagnation"] = -self.config.stagnation_penalty
                self._stagnation_count = 0
        info["stagnation_count"] = self._stagnation_count

        # --- Score reading (OCR) ---
        score_crop = self._crop_region(frame, self.config.score_region)
        current_score = self._ocr.read_number(score_crop)

        if current_score is not None:
            if self._prev_score is not None and current_score > self._prev_score:
                delta = current_score - self._prev_score
                score_reward = self.config.score_reward_multiplier * delta
                reward += score_reward
                breakdown["score"] = score_reward
            self._prev_score = current_score

        info["score"] = current_score
        info["reward_breakdown"] = breakdown

        return reward, info

    def _calculate_visual_novelty(self, frame: np.ndarray) -> float:
        """Estimate frame-to-frame scene change to reward exploration/movement."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32)

        if self._prev_novelty_frame is None:
            self._prev_novelty_frame = small
            return 0.0

        novelty = float(np.mean(np.abs(small - self._prev_novelty_frame)))
        self._prev_novelty_frame = small
        return novelty

    @staticmethod
    def _crop_region(
        frame: np.ndarray,
        region: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Crop a region from the frame."""
        height, width = frame.shape[:2]
        left, top, right, bottom = [int(value) for value in region]
        left = max(0, min(left, width - 1))
        right = max(left + 1, min(right, width))
        top = max(0, min(top, height - 1))
        bottom = max(top + 1, min(bottom, height))
        return frame[top:bottom, left:right]
