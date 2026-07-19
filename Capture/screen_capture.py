"""
MegaBonk AI — Screen Capture Module
====================================
High-performance screen capture using DXCam (Windows Desktop Duplication API).
Provides real-time frame capture at up to 60+ FPS with minimal CPU overhead.
"""

from __future__ import annotations

import time
import logging
from typing import Optional

import dxcam
import numpy as np

logger = logging.getLogger(__name__)


class ScreenCapture:
    """
    Wrapper around DXCam for high-speed screen capture.

    Supports:
    - Full-screen or region-based capture
    - Threaded capture at configurable FPS
    - Frame retrieval (latest or blocking)

    Usage:
        cap = ScreenCapture(target_fps=60, region=(0, 0, 1920, 1080))
        cap.start()
        frame = cap.get_latest_frame()  # numpy array (H, W, 3) BGR
        cap.stop()
    """

    def __init__(
        self,
        target_fps: int = 60,
        region: Optional[tuple[int, int, int, int]] = None,
        output_color: str = "BGR",
    ) -> None:
        """
        Initialize the screen capture.

        Args:
            target_fps: Target capture framerate.
            region: Capture region as (left, top, right, bottom).
                    None = capture entire primary monitor.
            output_color: Color format — "BGR" (for OpenCV) or "RGB".
        """
        self.target_fps = target_fps
        self.region = tuple(region) if region else None
        self.output_color = output_color

        self._camera: Optional[dxcam.DXCamera] = None
        self._running = False
        self._last_frame: Optional[np.ndarray] = None

        logger.info(
            "ScreenCapture initialized: fps=%d, region=%s, color=%s",
            target_fps,
            region,
            output_color,
        )

    def start(self) -> None:
        """Start the threaded capture loop."""
        if self._running:
            logger.warning("Capture already running, ignoring start()")
            return

        self._camera = dxcam.create(output_color=self.output_color)
        self._camera.start(
            target_fps=self.target_fps,
            region=self.region,
        )
        self._running = True
        logger.info("Screen capture started at %d FPS", self.target_fps)

    def stop(self) -> None:
        """Stop the capture loop and release the camera."""
        if not self._running:
            return

        if self._camera is not None:
            self._camera.stop()
            del self._camera
            self._camera = None

        self._running = False
        self._last_frame = None
        logger.info("Screen capture stopped")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent captured frame (non-blocking).

        Returns:
            numpy array of shape (H, W, 3) in the configured color format,
            or None if no frame is available yet.
        """
        if not self._running or self._camera is None:
            logger.warning("Capture not running, returning None")
            return None

        frame = self._camera.get_latest_frame()
        if frame is not None:
            self._last_frame = frame
        return self._last_frame

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the next NEW frame (blocking until a new frame arrives).

        Args:
            timeout: Maximum time (seconds) to wait for a new frame.

        Returns:
            numpy array of shape (H, W, 3), or None on timeout.
        """
        if not self._running or self._camera is None:
            logger.warning("Capture not running, returning None")
            return None

        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            frame = self._camera.get_latest_frame()
            if frame is not None:
                self._last_frame = frame
                return frame
            time.sleep(0.001)

        logger.warning("get_frame() timed out after %.2fs", timeout)
        return self._last_frame

    def grab_single(self) -> Optional[np.ndarray]:
        """
        Grab a single frame without starting the threaded loop.
        Useful for calibration / testing.

        Returns:
            numpy array of shape (H, W, 3), or None on failure.
        """
        camera = dxcam.create(output_color=self.output_color)
        frame = camera.grab(region=self.region)
        del camera
        return frame

    @property
    def is_running(self) -> bool:
        """Check if capture is currently active."""
        return self._running

    def __enter__(self) -> "ScreenCapture":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    def __del__(self) -> None:
        self.stop()
