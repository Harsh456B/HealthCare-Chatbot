# Medical Chatbot - Quick Start Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Medical Chatbot - Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Step 1: Activating virtual environment..." -ForegroundColor Green
& .\medibot\Scripts\Activate.ps1

Write-Host ""
Write-Host "Step 2: Starting Flask application..." -ForegroundColor Green
Write-Host ""
Write-Host "The chatbot will be available at: http://localhost:8080" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python app.py
