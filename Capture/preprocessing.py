"""
MegaBonk AI — Frame Preprocessing Module
==========================================
Handles frame resizing, grayscale conversion, normalization,
and frame stacking for the RL agent's observations.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np


class FramePreprocessor:
    """
    Preprocesses raw screen captures into agent-ready observations.

    Pipeline:
        Raw BGR frame (1920x1080x3)
        → Resize (84x84)
        → Grayscale (84x84x1)
        → Stack N frames (N x 84 x 84) — channels-first for CnnPolicy

    Usage:
        preprocessor = FramePreprocessor(width=84, height=84, grayscale=True, stack_size=4)
        preprocessor.reset()
        observation = preprocessor.process(frame)  # shape: (4, 84, 84)
    """

    def __init__(
        self,
        width: int = 84,
        height: int = 84,
        grayscale: bool = True,
        stack_size: int = 4,
    ) -> None:
        """
        Args:
            width: Target width for resized frames.
            height: Target height for resized frames.
            grayscale: Whether to convert frames to grayscale.
            stack_size: Number of frames to stack (temporal context).
        """
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.stack_size = stack_size

        # Frame stack — deque with fixed max length
        self._frame_stack: deque[np.ndarray] = deque(maxlen=stack_size)

    def reset(self) -> None:
        """Clear the frame stack. Call this at the start of each episode."""
        self._frame_stack.clear()

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a raw frame and return the stacked observation.

        Args:
            frame: Raw BGR frame from DXCam, shape (H, W, 3).

        Returns:
            Stacked observation, shape (stack_size, height, width), dtype uint8.
            Channels-first format for SB3 CnnPolicy.
        """
        processed = self._preprocess_single(frame)

        # Add to stack
        self._frame_stack.append(processed)

        # If stack isn't full yet, pad with copies of the first frame
        while len(self._frame_stack) < self.stack_size:
            self._frame_stack.appendleft(self._frame_stack[0].copy())

        # Stack into (N, H, W) array — channels-first
        stacked = np.stack(list(self._frame_stack), axis=0)
        return stacked

    def _preprocess_single(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame: resize + optional grayscale.

        Args:
            frame: Raw BGR frame, shape (H, W, 3).

        Returns:
            Preprocessed frame, shape (height, width), dtype uint8.
        """
        # Resize
        resized = cv2.resize(
            frame,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )

        # Convert to grayscale
        if self.grayscale:
            if len(resized.shape) == 3 and resized.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        return resized

    def get_current_stack(self) -> Optional[np.ndarray]:
        """
        Get the current frame stack without processing a new frame.

        Returns:
            Stacked observation or None if stack is empty.
        """
        if len(self._frame_stack) == 0:
            return None
        # Pad if needed
        while len(self._frame_stack) < self.stack_size:
            self._frame_stack.appendleft(self._frame_stack[0].copy())
        return np.stack(list(self._frame_stack), axis=0)

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Return the shape of the output observation: (stack_size, H, W)."""
        return (self.stack_size, self.height, self.width)


class RegionCropper:
    """
    Crops specific regions from a frame for OCR / template matching.

    Useful for extracting HUD elements (HP bar, XP bar, score counter).

    Usage:
        cropper = RegionCropper()
        hp_crop = cropper.crop(frame, region=(50, 50, 300, 80))
    """

    @staticmethod
    def crop(
        frame: np.ndarray,
        region: tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Crop a region from the frame.

        Args:
            frame: Full frame, shape (H, W, ...).
            region: (left, top, right, bottom) in pixel coordinates.

        Returns:
            Cropped sub-image.
        """
        left, top, right, bottom = region
        return frame[top:bottom, left:right].copy()

    @staticmethod
    def crop_multiple(
        frame: np.ndarray,
        regions: dict[str, tuple[int, int, int, int]],
    ) -> dict[str, np.ndarray]:
        """
        Crop multiple named regions from the frame.

        Args:
            frame: Full frame, shape (H, W, ...).
            regions: Dict mapping region names to (left, top, right, bottom).

        Returns:
            Dict mapping region names to cropped sub-images.
        """
        return {
            name: RegionCropper.crop(frame, region)
            for name, region in regions.items()
        }
