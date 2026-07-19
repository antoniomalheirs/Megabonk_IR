"""
MegaBonk AI — Training Callbacks
===================================
Custom callbacks for Stable-Baselines3 training:
- Periodic checkpointing with episode stats
- TensorBoard logging of custom game metrics
- Hotkey-based pause/resume functionality
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class GameMetricsCallback(BaseCallback):
    """
    Logs custom game metrics to TensorBoard.

    Tracks per-episode:
    - Episode length (steps survived)
    - Episode total reward
    - Average HP
    - Max score reached
    - Number of level ups
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._episode_hp_sum: float = 0.0
        self._episode_hp_count: int = 0
        self._episode_max_score: float = 0.0
        self._episode_level_ups: int = 0
        self._episode_count: int = 0

    def _on_step(self) -> bool:
        """Called after each step."""
        # Extract info from the most recent step
        infos = self.locals.get("infos", [])

        for info in infos:
            # Track HP
            hp = info.get("hp")
            if hp is not None:
                self._episode_hp_sum += hp
                self._episode_hp_count += 1

            # Track score
            score = info.get("score")
            if score is not None and score > self._episode_max_score:
                self._episode_max_score = score

            # Track level ups
            breakdown = info.get("reward_breakdown", {})
            if "level_up" in breakdown:
                self._episode_level_ups += 1

            # Check for episode end
            if info.get("dead", False) or info.get("TimeLimit.truncated", False):
                self._episode_count += 1
                ep_reward = info.get("episode_reward", 0.0)
                ep_length = info.get("episode_step", 0)

                # Log to TensorBoard
                self.logger.record("game/episode_reward", ep_reward)
                self.logger.record("game/episode_length", ep_length)
                self.logger.record("game/episode_count", self._episode_count)

                if self._episode_hp_count > 0:
                    avg_hp = self._episode_hp_sum / self._episode_hp_count
                    self.logger.record("game/avg_hp", avg_hp)

                self.logger.record("game/max_score", self._episode_max_score)
                self.logger.record("game/level_ups", self._episode_level_ups)

                if self.verbose > 0:
                    logger.info(
                        "Episode %d: steps=%d, reward=%.2f, max_score=%.0f, level_ups=%d",
                        self._episode_count,
                        ep_length,
                        ep_reward,
                        self._episode_max_score,
                        self._episode_level_ups,
                    )

                # Reset episode counters
                self._episode_hp_sum = 0.0
                self._episode_hp_count = 0
                self._episode_max_score = 0.0
                self._episode_level_ups = 0

        return True


class SmartCheckpointCallback(BaseCallback):
    """
    Saves model checkpoints periodically and on best performance.

    Saves:
    - Every N steps (periodic)
    - When a new best episode reward is achieved
    """

    def __init__(
        self,
        save_freq: int = 10_000,
        save_path: str = "Checkpoints",
        name_prefix: str = "megabonk_ppo",
        verbose: int = 1,
    ) -> None:
        """
        Args:
            save_freq: Save every N steps.
            save_path: Directory to save checkpoints.
            name_prefix: Prefix for checkpoint filenames.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self._best_reward = float("-inf")

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        # Periodic checkpoint
        if self.n_calls % self.save_freq == 0:
            path = self.save_path / f"{self.name_prefix}_{self.n_calls}_steps"
            self.model.save(str(path))
            if self.verbose > 0:
                logger.info("Checkpoint saved: %s", path)

        # Best reward checkpoint
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_reward = info.get("episode_reward")
            if ep_reward is not None and info.get("dead", False):
                if ep_reward > self._best_reward:
                    self._best_reward = ep_reward
                    path = self.save_path / f"{self.name_prefix}_best"
                    self.model.save(str(path))
                    if self.verbose > 0:
                        logger.info(
                            "New best reward: %.2f — saved to %s",
                            ep_reward,
                            path,
                        )

        return True


class TrainingTimerCallback(BaseCallback):
    """
    Tracks and logs training speed metrics.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self._start_time: float = 0.0
        self._last_log_time: float = 0.0
        self._last_log_steps: int = 0

    def _init_callback(self) -> None:
        self._start_time = time.time()
        self._last_log_time = self._start_time

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            now = time.time()
            elapsed = now - self._last_log_time
            steps_delta = self.n_calls - self._last_log_steps

            if elapsed > 0:
                fps = steps_delta / elapsed
                total_elapsed = now - self._start_time
                self.logger.record("time/fps", fps)
                self.logger.record("time/total_minutes", total_elapsed / 60)

                if self.verbose > 0:
                    logger.info(
                        "Step %d: %.1f FPS, %.1f min elapsed",
                        self.n_calls,
                        fps,
                        total_elapsed / 60,
                    )

            self._last_log_time = now
            self._last_log_steps = self.n_calls

        return True
