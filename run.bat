@echo off
echo ========================================================
echo   Starting GoTap Smart Student Transportation System
echo ========================================================
echo.
cd /d "%~dp0"
.\.venv\Scripts\python.exe manage.py runserver
pause
