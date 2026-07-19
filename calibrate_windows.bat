@echo off
setlocal
cd /d "%~dp0"
python run_reward_calibration.py --window-title "MegaBonk" --update-config --templates --skip-ocr-test
pause
