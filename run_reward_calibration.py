"""
MegaBonk AI — Reward HUD Calibration Tool
===========================================
Interactive tool to set the coordinates for HP, XP, and Score regions.

Usage:
1. Run this script. It will grab a frame from your screen.
2. The screen will freeze and display a window.
3. Use your mouse to click and drag a rectangle over the target region.
4. Press SPACE or ENTER to confirm the region, or 'c' to cancel/redo.
5. The selected coordinates will be printed so you can update your config!
"""

import sys
import yaml
import cv2
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from Capture.screen_capture import ScreenCapture
from Environment.rewards import OCRReader, BarReader


def select_region(image, window_name="Select Region (Enter to confirm, c to cancel)"):
    """Interactive ROI selection with OpenCV."""
    print(f"--- Selecting Region for {window_name} ---")
    print("Click and drag to draw a rectangle. Press SPACE or ENTER to confirm. Press 'c' to retry.")
    
    # Select ROI
    roi = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    
    # roi is (x, y, w, h)
    if roi[2] == 0 or roi[3] == 0:
        print("Selection cancelled.")
        return None
        
    x, y, w, h = roi
    # Return as [left, top, right, bottom]
    return [x, y, x + w, y + h]


def main():
    print("Starting calibration...")
    print("Make sure MegaBonk is visible on your main screen!")
    print("Grabbing frame in 3 seconds...")
    import time
    time.sleep(3)
    
    cap = ScreenCapture()
    frame = cap.grab_single()
    if frame is None:
        print("Failed to capture screen.")
        sys.exit(1)
        
    print(f"Captured frame shape: {frame.shape}")
    
    # Calibrate HP
    hp_region = select_region(frame, "HP Bar")
    if hp_region:
        print(f"HP Region: {hp_region}")
        hp_crop = frame[hp_region[1]:hp_region[3], hp_region[0]:hp_region[2]]
        pct = BarReader.read_hp_bar(hp_crop)
        print(f"Test HP Bar fill: {pct:.2%}")
        
    # Calibrate XP
    xp_region = select_region(frame, "XP Bar")
    if xp_region:
        print(f"XP Region: {xp_region}")
        xp_crop = frame[xp_region[1]:xp_region[3], xp_region[0]:xp_region[2]]
        pct = BarReader.read_xp_bar(xp_crop)
        print(f"Test XP Bar fill: {pct:.2%}")
        
    # Calibrate Score
    score_region = select_region(frame, "Score Counter")
    if score_region:
        print(f"Score Region: {score_region}")
        score_crop = frame[score_region[1]:score_region[3], score_region[0]:score_region[2]]
        ocr = OCRReader()
        val = ocr.read_number(score_crop)
        print(f"Test Score OCR: {val}")

    print("\n--- Summary for default.yaml ---")
    if hp_region: print(f"hp_region: {hp_region}")
    if xp_region: print(f"xp_region: {xp_region}")
    if score_region: print(f"score_region: {score_region}")
    print("Update Configs/default.yaml with these values.")

if __name__ == "__main__":
    main()
