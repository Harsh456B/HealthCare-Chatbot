@echo off
echo ========================================
echo   Medical Chatbot - Quick Start
echo ========================================
echo.

echo Step 1: Activating virtual environment...
call medibot\Scripts\activate.bat

echo.
echo Step 2: Starting Flask application...
echo.
echo The chatbot will be available at: http://localhost:8080
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
