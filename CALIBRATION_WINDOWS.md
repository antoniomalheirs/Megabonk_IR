# MegaBonk reward calibration on Windows

Use this tool to generate the capture and reward coordinates used by training and inference.

## 1. Install dependencies

From the repository folder:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Open MegaBonk

Start MegaBonk and leave the HUD visible in windowed or borderless mode. The default config looks for a window title containing `MegaBonk`.

If the title is different, list windows first:

```powershell
python run_reward_calibration.py --list-windows
```

## 3. Calibrate live capture and HUD regions

Recommended command:

```powershell
python run_reward_calibration.py --window-title "MegaBonk" --update-config --templates --skip-ocr-test
```

The tool will:

1. find and focus the MegaBonk window;
2. capture only that window and save it as `capture.region` in `Configs/default.yaml`;
3. ask you to drag rectangles around the HP bar, XP bar and score counter;
4. optionally ask for game-over and level-up templates when `--templates` is used;
5. save an annotated preview at `Configs/calibration_preview.png`;
6. create `Configs/default.yaml.bak` before writing changes.

Coordinates selected in the OpenCV windows are relative to the captured game frame, so they match the frames used by the environment after the capture region is saved.

## Useful alternatives

Use a screenshot instead of live DXCam capture:

```powershell
python run_reward_calibration.py --image .\my_screenshot.png --update-config --templates --skip-ocr-test
```

Capture a manually chosen screen rectangle:

```powershell
python run_reward_calibration.py --region 100 80 1700 980 --update-config --templates --skip-ocr-test
```

Capture the full screen instead of the game window/configured region:

```powershell
python run_reward_calibration.py --no-window-region --update-config --templates --skip-ocr-test
```

## Controls during selection

- Drag with the mouse to draw a rectangle.
- Press `Space` or `Enter` to accept it.
- Press `c` to skip the current region.

After calibration, inspect `Configs/calibration_preview.png`. If a rectangle is wrong, rerun the command; the previous YAML is backed up before each write.

## Fast HUD-only setup

For the easiest HUD configuration, run:

```powershell
configure_hud_windows.bat
```

This is equivalent to:

```powershell
python run_reward_calibration.py --window-title "MegaBonk" --quick-hud --preview Configs\hud_preview.png
```

It only asks for HP, XP and score boxes, updates `Configs/default.yaml`, skips the slow OCR model startup, and saves `Configs/hud_preview.png`.

To review the boxes currently saved in the YAML without changing anything:

```powershell
python run_reward_calibration.py --window-title "MegaBonk" --review-current --preview Configs\hud_review.png
```

To fix only one region:

```powershell
python run_reward_calibration.py --window-title "MegaBonk" --quick-hud --regions score
```

## After calibration: train the model

With MegaBonk still open, start training from the repository folder:

```powershell
python Trainer\train.py --config Configs\default.yaml
```

The training script reads `Configs/default.yaml`, creates `MegaBonkEnv`, trains PPO, writes checkpoints to `Checkpoints`, writes TensorBoard logs to `Logs`, and saves the final model under `Models`.

To resume from a checkpoint:

```powershell
python Trainer\train.py --config Configs\default.yaml --resume Checkpoints\megabonk_ppo_10000_steps.zip
```

## Skill, perk, and map item handling

The agent action space includes an explicit interact action. By default, `actions.interact_key` is `e`, so the policy can learn to press `E` for pickups, map items, button totems, proximity prompts, or interaction prompts.

The environment also has safety nets for level-up/perk screens and interactable map objects:

```yaml
environment:
  auto_confirm_level_up: true
  auto_confirm_key: "enter"
  auto_confirm_cooldown_steps: 30
  auto_confirm_repeats: 2
  auto_interact_enabled: true
  auto_interact_key: "e"
  auto_interact_every_steps: 45
  auto_interact_hold_duration: 0.05
```

When the reward detector sees a level-up/perk-choice template or XP reset, the environment schedules confirm key presses, respecting the cooldown. This prevents training from freezing on a perk screen. If your game confirms perks with a different key, change `auto_confirm_key` in `Configs/default.yaml`.

For totems and pickups, the environment periodically taps `auto_interact_key`. This makes the character try button totems and nearby/proximity interactables even before the policy has learned perfect timing. The policy still receives the explicit interact action and can learn to use it directly.

If you want the neural network to learn all perk menu navigation and totem timing manually, set `auto_confirm_level_up: false` and `auto_interact_enabled: false`, then train a fresh model.

## Run the trained model

```powershell
python Inference\play.py --config Configs\default.yaml
```

Or choose a model manually:

```powershell
python Inference\play.py --config Configs\default.yaml --model Models\megabonk_ppo.zip
```

## Reward tuning for correct Megabonk behavior

The default rewards are designed for Megabonk's survivor-like loop: survive, keep moving, collect XP, level up, choose perks/skills, kill enemies, and interact with map objects.

Important values in `Configs/default.yaml`:

```yaml
rewards:
  survival_reward: 0.01
  xp_reward: 5.0
  hp_loss_penalty: 1.0
  death_penalty: -10.0
  score_reward_multiplier: 0.1
  exploration_reward: 0.002
  visual_novelty_threshold: 6.0
  stagnation_penalty: 0.01
  stagnation_steps: 90
```

Use these tuning rules:

- If the character stands still, increase `exploration_reward` slightly or lower `visual_novelty_threshold`.
- If the character runs around but ignores survival, lower `exploration_reward` or increase `hp_loss_penalty`.
- If the character misses button totems, lower `environment.auto_interact_every_steps` so it taps interact more often.
- If perk choices stay open, increase `environment.auto_confirm_repeats` or reduce `environment.auto_confirm_cooldown_steps`.
- If score OCR is noisy, reduce `score_reward_multiplier` and rely more on XP/survival/exploration rewards.
