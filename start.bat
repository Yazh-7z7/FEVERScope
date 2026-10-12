@echo off
REM FEVERScope Unified Launcher (Windows)
REM Starts FastAPI backend (port 8000) and React frontend (port 5173)

echo ==========================================================
echo  Starting FEVERScope Backend and Frontend on Localhost
echo ==========================================================

start "FEVERScope Backend (FastAPI)" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

start "FEVERScope Frontend (React)" cmd /k "npm --prefix frontend run dev"

echo.
echo FEVERScope services launched in separate windows!
echo   Frontend: http://localhost:5173
echo   Backend API: http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.
pause
