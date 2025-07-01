#!/bin/bash

# Optional argument for virtual environment path (default: .venv)
VENV_PATH=${1:-.venv}

# Store PIDs to kill later
PIDS=()

# Handle Ctrl+C (SIGINT)
cleanup() {
  echo "Stopping processes..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null
  done
  exit 0
}

trap cleanup SIGINT

# Start frontend
echo "Starting frontend..."
(
  cd frontend || exit 1
  npm run dev
) &
PIDS+=($!)

# Start backend
echo "Starting backend with venv: $VENV_PATH"
(
  source "$VENV_PATH/bin/activate" || exit 1
  python -m server
) &
PIDS+=($!)

# Wait for both processes
wait