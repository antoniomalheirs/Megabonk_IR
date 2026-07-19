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

The agent action space includes an explicit interact action. By default, `actions.interact_key` is `e`, so the policy can learn to press `E` for pickups, map items, or interaction prompts.

The environment also has an auto-confirm safety net for level-up/perk screens:

```yaml
environment:
  auto_confirm_level_up: true
  auto_confirm_key: "enter"
  auto_confirm_cooldown_steps: 30
```

When the reward detector sees a level-up/perk-choice template or XP reset, the environment presses the confirm key once, respecting the cooldown. This prevents training from freezing on a perk screen. If your game confirms perks with a different key, change `auto_confirm_key` in `Configs/default.yaml`.

If you want the neural network to learn all perk menu navigation manually instead of auto-confirming the highlighted choice, set `auto_confirm_level_up: false` and add the needed menu keys to the action controller before training a new model.

## Run the trained model

```powershell
python Inference\play.py --config Configs\default.yaml
```

Or choose a model manually:

```powershell
python Inference\play.py --config Configs\default.yaml --model Models\megabonk_ppo.zip
```
