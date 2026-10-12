#!/usr/bin/env bash
# FEVERScope Unified Launcher (macOS / Linux)
# Starts FastAPI backend (port 8000) and React frontend (port 5173)

set -e

echo "=========================================================="
echo " Starting FEVERScope Backend & Frontend on Localhost"
echo "=========================================================="

cleanup() {
  echo ""
  echo "Shutting down FEVERScope services..."
  kill $(jobs -p) 2>/dev/null || true
  exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# 1. Start FastAPI backend
echo "-> Starting FastAPI Backend on http://localhost:8000..."
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# 2. Start React frontend
echo "-> Starting React Frontend on http://localhost:5173..."
npm --prefix frontend run dev &
FRONTEND_PID=$!

echo ""
echo "FEVERScope is running!"
echo "  Frontend: http://localhost:5173"
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo "Press Ctrl+C to stop all services."
echo ""

wait
