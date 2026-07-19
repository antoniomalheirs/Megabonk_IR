"""
MegaBonk AI — Play Script
===========================
Watch the trained AI play the game in real-time.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml
from stable_baselines3 import PPO

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from Environment.megabonk_env import MegaBonkEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("play")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch MegaBonk AI Play")
    parser.add_argument(
        "--config",
        type=str,
        default=str(project_root / "Configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override path to trained model (.zip)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    inf_cfg = config.get("inference", {})
    model_path = args.model or inf_cfg.get("model_path", "Models/megabonk_ppo.zip")

    logger.info("Loading model from %s", model_path)
    
    # Initialize environment
    logger.info("Initializing environment...")
    env = MegaBonkEnv(config=config, render_mode="human" if inf_cfg.get("show_overlay", True) else None)
    
    # Load model
    try:
        model = PPO.load(model_path, env=env, device=config.get("ppo", {}).get("device", "cuda"))
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        env.close()
        sys.exit(1)

    logger.info("Starting inference loop. Press Ctrl+C in terminal to stop.")
    
    try:
        obs, info = env.reset()
        ep_reward = 0.0
        
        while True:
            # Predict action (deterministic=True for evaluation)
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            
            # Optional: custom render handling can be added here
            env.render()
            
            if terminated or truncated:
                logger.info("Episode finished! Total Reward: %.2f", ep_reward)
                obs, info = env.reset()
                ep_reward = 0.0
                time.sleep(1.0)
                
    except KeyboardInterrupt:
        logger.info("Inference stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
