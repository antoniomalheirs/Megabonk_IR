@echo off
setlocal
cd /d "%~dp0"
python run_reward_calibration.py --window-title "MegaBonk" --quick-hud --preview "Configs\hud_preview.png"
pause
