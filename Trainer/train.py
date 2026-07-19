"""
MegaBonk AI — Training Script
===============================
Main script to start or resume training the PPO agent on MegaBonk.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_checker import check_env

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from Environment.megabonk_env import MegaBonkEnv
from Trainer.callbacks import (
    GameMetricsCallback,
    SmartCheckpointCallback,
    TrainingTimerCallback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MegaBonk AI")
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "Configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint model to resume training from (.zip)",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Run SB3 env checker before training",
    )
    args = parser.parse_args()

    # 1. Load config
    logger.info("Loading config from %s", args.config)
    config = load_config(args.config)
    ppo_cfg = config.get("ppo", {})
    train_cfg = config.get("training", {})

    # 2. Create environment
    logger.info("Creating MegaBonkEnv...")
    env = MegaBonkEnv(config=config, render_mode="human")

    # Optional: verify environment follows API
    if args.check_env:
        logger.info("Checking environment (this will open the capture window)...")
        check_env(env, warn=True)
        logger.info("Environment check passed!")

    # 3. Setup Callbacks
    callbacks = CallbackList(
        [
            GameMetricsCallback(verbose=0),
            SmartCheckpointCallback(
                save_freq=train_cfg.get("checkpoint_freq", 10_000),
                save_path=train_cfg.get("checkpoint_dir", "Checkpoints"),
                name_prefix="megabonk_ppo",
                verbose=1,
            ),
            TrainingTimerCallback(log_freq=1000, verbose=1),
        ]
    )

    # 4. Initialize or load Model
    if args.resume:
        logger.info("Resuming training from %s", args.resume)
        model = PPO.load(
            args.resume,
            env=env,
            device=ppo_cfg.get("device", "cuda"),
            tensorboard_log=train_cfg.get("log_dir", "Logs"),
        )
    else:
        logger.info("Initializing new PPO model (%s)...", ppo_cfg.get("policy", "CnnPolicy"))
        model = PPO(
            ppo_cfg.get("policy", "CnnPolicy"),
            env,
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            n_steps=ppo_cfg.get("n_steps", 2048),
            batch_size=ppo_cfg.get("batch_size", 64),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            ent_coef=ppo_cfg.get("ent_coef", 0.01),
            vf_coef=ppo_cfg.get("vf_coef", 0.5),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            verbose=ppo_cfg.get("verbose", 1),
            device=ppo_cfg.get("device", "cuda"),
            tensorboard_log=train_cfg.get("log_dir", "Logs"),
        )

    # 5. Train
    total_timesteps = train_cfg.get("total_timesteps", 500_000)
    logger.info("Starting training for %d timesteps...", total_timesteps)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=not bool(args.resume),
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving final model...")
    finally:
        # Save final model
        save_path = str(project_root / train_cfg.get("model_save_path", "Models/megabonk_ppo"))
        logger.info("Saving model to %s", save_path)
        model.save(save_path)
        env.close()

if __name__ == "__main__":
    main()
