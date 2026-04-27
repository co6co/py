@echo off
chcp 65001 >nul
echo ========================================
echo   WebRTC 信令服务器启动脚本
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.7+
    pause
    exit /b 1
)

echo [信息] 正在启动 WebRTC 信令服务器...
echo.

REM 启动服务器
python webrtc_server_for_http.py

pause
