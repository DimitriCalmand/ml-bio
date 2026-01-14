#!/bin/bash
# Get the directory where the script is located to ensure relative paths work
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if python3 is available, otherwise try python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

echo "Running training script from $SCRIPT_DIR..."
$PYTHON_CMD src/train.py