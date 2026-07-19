# MegaBonk IR — training an AI agent for MegaBonk

This repository wraps MegaBonk as a Gymnasium environment for reinforcement learning.
It captures the game window with DXCam, preprocesses frames, sends keyboard/mouse input with PyDirectInput, calculates rewards from the HUD, and trains a PPO agent with Stable-Baselines3.

> **Windows only for live gameplay.** DXCam and PyDirectInput are intended for the Windows desktop. Keep MegaBonk open and visible while calibrating, training, or running inference.

## What the agent can control

The action space is configured as:

```text
MultiDiscrete([3, 3, 2, 9, 7, 2])
```

The six action components are:

1. horizontal movement: nothing / `A` / `D`;
2. vertical movement: nothing / `W` / `S`;
3. jump: nothing / `Space`;
4. mouse X delta;
5. mouse Y delta;
6. interact/confirm: nothing / configured interact key, default `E`.

That last interact action lets the policy learn to pick up map items and interact with prompts. In addition, the environment can automatically press `Enter` when a level-up/perk choice is detected so training does not stall on choice screens.

## 1. Install the project on Windows

Open PowerShell in the repository folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If you do not have CUDA/NVIDIA working, edit `Configs/default.yaml` and change:

```yaml
ppo:
  device: "cpu"
```

If CUDA is working, keep `device: "cuda"`.

## 2. Open MegaBonk

Start MegaBonk before calibration. Windowed or borderless mode is recommended because the calibration tool can find and capture the game window by title.

If the game title is not exactly `MegaBonk`, list visible windows:

```powershell
python run_reward_calibration.py --list-windows
```

Use the title text that appears in the list with `--window-title`.

## 3. Calibrate capture, HUD rewards, perks, and templates

Recommended command:

```powershell
python run_reward_calibration.py --window-title "MegaBonk" --update-config --templates --skip-ocr-test
```

Or run the convenience launcher:

```powershell
calibrate_windows.bat
```

During calibration:

1. MegaBonk is focused and captured.
2. Select the HP bar rectangle.
3. Select the XP bar rectangle.
4. Select the score rectangle.
5. If `--templates` is enabled, select a game-over template and a level-up/perk-choice template.
6. The tool writes the chosen capture region and reward regions to `Configs/default.yaml`.
7. The tool saves `Configs/calibration_preview.png` so you can verify the rectangles.
8. The old YAML is backed up as `Configs/default.yaml.bak`.

Open the preview after calibration. If the rectangles are wrong, run calibration again.

### Selection controls

- Drag with the mouse to draw a rectangle.
- Press `Space` or `Enter` to confirm.
- Press `c` to skip/retry the current item.

## 4. Configure skill/perk/item handling

The default config gives the AI an explicit interact action:

```yaml
actions:
  interact_options: 2
  interact_key: "e"
```

Use the key your MegaBonk build expects for picking up map items or interacting with prompts. If your game uses another key, change `interact_key`.

The default environment also auto-confirms level-up/perk screens:

```yaml
environment:
  auto_confirm_level_up: true
  auto_confirm_key: "enter"
  auto_confirm_cooldown_steps: 30
```

Use `auto_confirm_key` for the key that accepts the highlighted/default perk choice. This guarantees training continues through perk/skill choice screens instead of waiting forever. If you want the policy to learn all menu navigation manually, set `auto_confirm_level_up: false` and expose the needed keys through the action controller.

## 5. Optional environment check

With MegaBonk open and calibrated:

```powershell
python Trainer\train.py --config Configs\default.yaml --check-env
```

This verifies the Gymnasium interface, but it can send real inputs to the game.

## 6. Start training

```powershell
python Trainer\train.py --config Configs\default.yaml
```

The default config trains for `500_000` timesteps, saves checkpoints every `10_000` steps under `Checkpoints`, logs TensorBoard files under `Logs`, and saves the final model to `Models/megabonk_ppo.zip`.

Stop safely with `Ctrl+C`; the training script saves the model in its shutdown path.

## 7. Monitor training

In a second PowerShell with the venv active:

```powershell
tensorboard --logdir Logs
```

Open the URL TensorBoard prints, usually `http://localhost:6006`.

## 8. Resume from a checkpoint

```powershell
python Trainer\train.py --config Configs\default.yaml --resume Checkpoints\megabonk_ppo_10000_steps.zip
```

Replace the checkpoint path with the newest `.zip` in `Checkpoints`.

## 9. Run the trained model

```powershell
python Inference\play.py --config Configs\default.yaml
```

Or choose a model manually:

```powershell
python Inference\play.py --config Configs\default.yaml --model Models\megabonk_ppo.zip
```

## Troubleshooting

### The capture is black or wrong

- Run calibration again.
- Try windowed/borderless mode.
- Try `--no-window-region` to capture the full screen.
- Try an explicit region: `python run_reward_calibration.py --region LEFT TOP RIGHT BOTTOM --update-config --templates --skip-ocr-test`.

### The AI does not pick up items

- Confirm the game pickup/interact key.
- Set `actions.interact_key` to that key.
- Keep `interact_options: 2` so the action space includes the interact action.
- Train a fresh model after changing the action space.

### Training stops on perk/skill selection

- Calibrate `level_up_template` with `--templates`.
- Keep `environment.auto_confirm_level_up: true`.
- Set `environment.auto_confirm_key` to the key that accepts the highlighted perk.

### CUDA errors

Change `ppo.device` to `cpu` in `Configs/default.yaml`.
