#!/bin/bash
# Setup script to install Orange3 packages for web backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== Orange3 Web Backend Setup ==="
echo "Workspace root: $WORKSPACE_ROOT"

# Activate virtual environment if exists
if [ -d "$SCRIPT_DIR/.venv" ]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Install base requirements
echo "Installing base requirements..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# Install Orange3 (includes orange-canvas-core and orange-widget-base as dependencies)
echo "Installing orange3..."
pip install -e "$WORKSPACE_ROOT/orange3"

echo ""
echo "=== Setup Complete ==="
echo "Orange3 packages are now available for the web backend."
echo "(orange-canvas-core and orange-widget-base are installed as dependencies)"
echo ""
echo "To run the backend:"
echo "  cd $SCRIPT_DIR"
echo "  uvicorn app.main:app --reload --port 8000"
