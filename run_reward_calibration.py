"""
MegaBonk AI — Reward HUD Calibration Tool
===========================================
Interactive tool to set the coordinates for HP, XP, Score and optional
state-detection templates used by the reward calculator.

Typical usage:
    python run_reward_calibration.py --update-config

The tool can also reuse an existing screenshot for offline calibration:
    python run_reward_calibration.py --image capture.png --update-config
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import yaml

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from Capture.screen_capture import ScreenCapture
from Capture.window_utils import find_window, focus_window, list_windows
from Environment.rewards import BarReader, OCRReader

Region = list[int]

HUD_TARGETS = {
    "hp_region": "HP Bar",
    "xp_region": "XP Bar",
    "score_region": "Score Counter",
}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict for blank files."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(path: Path, data: dict[str, Any], make_backup: bool = True) -> None:
    """Write YAML with stable key ordering and readable block style."""
    if make_backup and path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"Backup saved: {backup}")
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def validate_region(region: Iterable[int], image_shape: tuple[int, ...]) -> Region:
    """Clamp and validate [left, top, right, bottom] coordinates."""
    height, width = image_shape[:2]
    left, top, right, bottom = [int(v) for v in region]
    left = max(0, min(left, width - 1))
    right = max(left + 1, min(right, width))
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))
    return [left, top, right, bottom]


def crop_region(image, region: Region):
    """Crop [left, top, right, bottom] from an image."""
    left, top, right, bottom = region
    return image[top:bottom, left:right]


def select_region(image, label: str) -> Region | None:
    """Interactive ROI selection with OpenCV."""
    window_name = f"Select {label} (Enter/Space confirm, c cancel)"
    print(f"\n--- Selecting {label} ---")
    print("Click and drag to draw a rectangle. Press SPACE/ENTER to confirm or 'c' to skip.")

    roi = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    if roi[2] == 0 or roi[3] == 0:
        print(f"{label}: skipped.")
        return None

    x, y, w, h = roi
    return validate_region([x, y, x + w, y + h], image.shape)


def capture_frame(delay: float, output_color: str, region: Region | None):
    """Capture one frame from the primary display after a delay."""
    print("Starting calibration...")
    print("Make sure MegaBonk is visible on your main screen!")
    if region:
        print(f"Capturing screen region: {region}")
    if delay > 0:
        print(f"Grabbing frame in {delay:.1f} seconds...")
        time.sleep(delay)

    cap = ScreenCapture(output_color=output_color, region=tuple(region) if region else None)
    frame = cap.grab_single()
    if frame is None:
        raise RuntimeError("Failed to capture screen. Try --image with a saved screenshot.")
    return frame


def resolve_capture_region(args: argparse.Namespace, config: dict[str, Any]) -> Region | None:
    """Resolve the live capture region from CLI, config or a Windows game window."""
    if args.image:
        return None
    if args.region:
        return validate_region(args.region, (99999, 99999, 3))
    window_title = args.window_title or config.get("capture", {}).get("window_title")
    if window_title and args.use_window_region:
        window = find_window(window_title)
        focus_window(window.hwnd)
        print(f"Found game window: {window.title!r} at {list(window.rect)}")
        return list(window.rect)
    configured = config.get("capture", {}).get("region")
    return list(configured) if configured else None


def load_frame(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Any, Region | None]:
    """Load the calibration frame from disk or live capture, returning the capture region used."""
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise RuntimeError(f"Failed to read image: {args.image}")
        return frame, None
    region = resolve_capture_region(args, config)
    return capture_frame(args.delay, args.output_color, region), region


def draw_preview(image, regions: dict[str, Region]):
    """Return an annotated copy showing all selected regions."""
    colors = {
        "hp_region": (0, 255, 0),
        "xp_region": (255, 128, 0),
        "score_region": (0, 255, 255),
    }
    preview = image.copy()
    for name, region in regions.items():
        left, top, right, bottom = region
        color = colors.get(name, (255, 255, 255))
        cv2.rectangle(preview, (left, top), (right, bottom), color, 2)
        cv2.putText(
            preview,
            name,
            (left, max(15, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return preview


def save_template(image, region: Region, output_path: Path) -> str:
    """Save a cropped template and return its project-relative path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop_region(image, region))
    return str(output_path.relative_to(project_root)).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate MegaBonk reward HUD regions")
    parser.add_argument("--config", type=Path, default=project_root / "Configs" / "default.yaml")
    parser.add_argument("--image", type=Path, help="Use an existing screenshot instead of live capture")
    parser.add_argument("--update-config", action="store_true", help="Write selected values into the YAML config")
    parser.add_argument(
        "--quick-hud",
        action="store_true",
        help="Shortcut for HUD-only setup: update config, skip OCR model load, no templates",
    )
    parser.add_argument(
        "--review-current",
        action="store_true",
        help="Draw current configured HUD regions on a fresh frame and exit",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=["hp", "xp", "score"],
        help="Calibrate only selected HUD regions instead of all three",
    )
    parser.add_argument("--window-title", help="Window title text to locate the MegaBonk window")
    parser.add_argument(
        "--use-window-region",
        action="store_true",
        default=True,
        help="Capture only the configured/window-title game window",
    )
    parser.add_argument(
        "--no-window-region",
        dest="use_window_region",
        action="store_false",
        help="Ignore window lookup and capture full screen/config region",
    )
    parser.add_argument(
        "--region",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="Explicit screen capture region",
    )
    parser.add_argument("--list-windows", action="store_true", help="List visible Windows titles and exit")
    parser.add_argument("--skip-ocr-test", action="store_true", help="Do not initialize EasyOCR while calibrating the score box")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds to wait before live capture")
    parser.add_argument("--output-color", default="BGR", choices=["BGR", "RGB"], help="DXCam output color")
    parser.add_argument("--preview", type=Path, default=project_root / "Configs" / "calibration_preview.png")
    parser.add_argument("--templates", action="store_true", help="Also select and save game-over/level-up templates")
    args = parser.parse_args()

    if args.quick_hud:
        args.update_config = True
        args.skip_ocr_test = True
        args.templates = False

    if args.list_windows:
        for window in list_windows():
            print(f"{window.hwnd}: {window.title} {list(window.rect)}")
        return

    config = load_yaml(args.config) if args.config.exists() else {}
    frame, capture_region = load_frame(args, config)
    print(f"Captured frame shape: {frame.shape}")

    if args.review_current:
        rewards = config.get("rewards", {})
        current_regions = {
            key: list(rewards[key])
            for key in HUD_TARGETS
            if key in rewards and rewards[key]
        }
        if not current_regions:
            print("No HUD regions found in config to review.")
            return
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), draw_preview(frame, current_regions))
        print(f"Saved current HUD preview: {args.preview}")
        return

    selected: dict[str, Region] = {}
    wanted = set(args.regions or ["hp", "xp", "score"])
    region_key_to_short = {
        "hp_region": "hp",
        "xp_region": "xp",
        "score_region": "score",
    }
    for key, label in HUD_TARGETS.items():
        if region_key_to_short[key] not in wanted:
            continue
        region = select_region(frame, label)
        if region is None:
            continue
        selected[key] = region
        print(f"{key}: {region}")
        crop = crop_region(frame, region)
        if key == "hp_region":
            print(f"Test HP Bar fill: {BarReader.read_hp_bar(crop):.2%}")
        elif key == "xp_region":
            print(f"Test XP Bar fill: {BarReader.read_xp_bar(crop):.2%}")
        elif args.skip_ocr_test:
            print("Test Score OCR: skipped (--skip-ocr-test)")
        else:
            print(f"Test Score OCR: {OCRReader().read_number(crop)}")

    templates: dict[str, str] = {}
    if args.templates:
        for key, label, filename in [
            ("game_over_template", "Game Over Template", "game_over.png"),
            ("level_up_template", "Level Up Template", "level_up.png"),
        ]:
            region = select_region(frame, label)
            if region is not None:
                templates[key] = save_template(
                    frame,
                    region,
                    project_root / "Configs" / "templates" / filename,
                )
                print(f"{key}: {templates[key]}")

    if selected:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), draw_preview(frame, selected))
        print(f"Saved annotated preview: {args.preview}")

    print("\n--- Summary for default.yaml ---")
    for key, value in selected.items():
        print(f"{key}: {value}")
    for key, value in templates.items():
        print(f"{key}: {value}")

    if args.update_config:
        capture_cfg = config.setdefault("capture", {})
        if capture_region is not None:
            capture_cfg["region"] = capture_region
        if args.window_title:
            capture_cfg["window_title"] = args.window_title
        rewards = config.setdefault("rewards", {})
        rewards.update(selected)
        rewards.update(templates)
        save_yaml(args.config, config)
        print(f"Updated config: {args.config}")
    else:
        print("Run again with --update-config to write these values automatically.")


if __name__ == "__main__":
    main()
