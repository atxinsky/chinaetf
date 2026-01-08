@echo off
chcp 65001 >nul
echo ETF量化回测系统启动中...
echo 访问地址: http://localhost:8505
echo.
python run.py web
pause
