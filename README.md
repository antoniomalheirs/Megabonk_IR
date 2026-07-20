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

That last interact action lets the policy learn to pick up map items, button totems, proximity prompts, and other interactions. In addition, the environment can automatically press `Enter` when a level-up/perk choice is detected so training does not stall on choice screens, and it can periodically press the interact key as a safety net while the policy is still learning.

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

### Easier HUD-only setup

If you only want to configure the HUD used by rewards, use the dedicated quick command:

```powershell
configure_hud_windows.bat
```

Or run it manually:

```powershell
python run_reward_calibration.py --window-title "MegaBonk" --quick-hud --preview Configs\hud_preview.png
```

Useful HUD shortcuts:

```powershell
# Review the HUD boxes currently saved in Configs/default.yaml
python run_reward_calibration.py --window-title "MegaBonk" --review-current --preview Configs\hud_review.png

# Recalibrate only one or two boxes instead of all HUD regions
python run_reward_calibration.py --window-title "MegaBonk" --quick-hud --regions hp xp
python run_reward_calibration.py --window-title "MegaBonk" --quick-hud --regions score
```

`--quick-hud` writes the selected HUD regions to `Configs/default.yaml`, skips the slow EasyOCR score test, avoids template selection, and saves an annotated preview so you can immediately see whether HP/XP/score are correct.

### Selection controls

- Drag with the mouse to draw a rectangle.
- Press `Space` or `Enter` to confirm.
- Press `c` to skip/retry the current item.


## Interface recognition for smarter decisions

The environment now exposes a dedicated UI-recognition layer to the policy. In addition to the downscaled gameplay frames, the observation can include semantic UI channels configured with:

```yaml
preprocessing:
  semantic_ui_channels: 3

ui:
  enabled: true
  templates:
    perk_choice: "Configs/templates/level_up.png"
    interact_prompt: "Configs/templates/interact_prompt.png"
    main_menu: "Configs/templates/main_menu.png"
    stage_select: "Configs/templates/stage_select.png"
    pause_menu: "Configs/templates/pause_menu.png"
    loading_screen: "Configs/templates/loading_screen.png"
```

These channels encode HP, XP, score, perk/level-up screens, menus, death state, and interactable prompts as machine-readable planes appended to the CNN input. That means the model does not need to infer every important UI state from tiny 84×84 pixels alone. The same recognizer is also logged under `info["ui"]` at every environment step so you can debug exactly what the agent thinks is on screen.

For best results, capture small template images from your own MegaBonk resolution/language and save them under `Configs/templates/`. The defaults are safe: missing optional templates are skipped with warnings, while the already-calibrated `level_up.png` can immediately drive perk-choice recognition.

## 4. Configure skill/perk/item handling

The default config gives the AI an explicit interact action:

```yaml
actions:
  interact_options: 2
  interact_key: "e"
```

Use the key your MegaBonk build expects for picking up map items or interacting with prompts. If your game uses another key, change `interact_key`.

The default environment also auto-confirms level-up/perk screens and periodically taps interact for button/proximity totems:

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

Use `auto_confirm_key` for the key that accepts the highlighted/default perk choice. `auto_confirm_repeats` presses it more than once, with `auto_confirm_cooldown_steps` between presses, because some level-up screens need a small delay before accepting input. This keeps training moving through perk/skill choice screens instead of waiting forever.

Use `auto_interact_key` for the key that activates button totems, pickup prompts, or interactable map objects. `auto_interact_every_steps` periodically taps that key while the agent is moving around, so proximity interactions and button totems are attempted even early in training. The neural network still has its own explicit interact action and can learn to press it at better times. If you want the policy to learn all menu and interaction timing manually, set `auto_confirm_level_up: false` and `auto_interact_enabled: false`.

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
- Set `actions.interact_key` and `environment.auto_interact_key` to that key.
- Keep `interact_options: 2` so the action space includes the interact action.
- Keep `environment.auto_interact_enabled: true` for button totems/proximity prompts during early training.
- Lower `environment.auto_interact_every_steps` if it is not trying often enough.
- Train a fresh model after changing the action space.

### Training stops on perk/skill selection

- Calibrate `level_up_template` with `--templates`.
- Keep `environment.auto_confirm_level_up: true`.
- Set `environment.auto_confirm_key` to the key that accepts the highlighted perk.
- Increase `environment.auto_confirm_repeats` if the game sometimes ignores the first key press.
- Lower `environment.auto_confirm_cooldown_steps` if the choice screen remains open too long.

### CUDA errors

Change `ppo.device` to `cpu` in `Configs/default.yaml`.

## Reward design notes

Megabonk plays like a 3D survivor/roguelite: the character moves through generated maps, attacks automatically, kills waves/bosses, collects XP, levels up into random upgrade/perk choices, and finds items, shrines, totems, chests, and other map interactables. The reward setup follows that loop:

- `survival_reward` gives a tiny reward each step for staying alive.
- `hp_loss_penalty` and `death_penalty` punish taking damage and dying.
- `xp_reward` rewards level-up events detected by XP-bar resets or the calibrated level-up/perk template.
- `score_reward_multiplier` rewards score increases, which approximates killing enemies and progressing combat.
- `exploration_reward` rewards visual novelty so the agent does not learn to stand still; moving around is important for XP gems, items, shrines, totems, and map objectives.
- `stagnation_penalty` punishes long periods with little visual change, helping avoid getting stuck against terrain or idling in one spot.
- `auto_interact_enabled` periodically taps the interact key for button totems and proximity/button prompts while the policy is still learning.
- `auto_confirm_level_up` keeps the run moving through perk/skill selection screens.

The shaping rewards are intentionally small. The main objectives remain survival, avoiding damage, gaining levels, killing enemies/raising score, and continuing the run.

If the agent explores too randomly, lower `exploration_reward`. If it stands still or misses too many map objects, raise `exploration_reward`, lower `visual_novelty_threshold`, or lower `auto_interact_every_steps`.
