@echo off
title MiroFish
echo ==============================================
echo 🚀 MiroFish Startup Script for Windows 🚀
echo ==============================================

:: Navigate to the directory of this script
cd /d "%~dp0"

echo [1/2] Starting Backend Server...
:: Using the specified local .venv python and setting FLASK_DEBUG=False
start "MiroFish Backend" cmd /k "cd backend && set FLASK_DEBUG=False && .\.venv\Scripts\python.exe -B run.py"

echo [2/2] Starting Frontend Server...
start "MiroFish Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ✅ Services are starting in separate windows.
echo 🌐 Frontend will be available at: http://localhost:5173 (khởi động 1 lúc sẽ tự mở)
echo 🔌 Backend API will be running at: http://localhost:5001
echo.
echo Đóng command window này bất cứ lúc nào (server vẫn chạy ở 2 cửa sổ cmd mới).
echo.
pause
