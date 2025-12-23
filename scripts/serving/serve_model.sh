#!/bin/bash
# Serve FabMi model using LLaMA-Factory API

set -e

MODEL_PATH=${1:-"../../models/ernie_semi_rca_merged"}
PORT=${2:-8000}

echo "=========================================="
echo "FabMi: Starting Model Server"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    exit 1
fi

# Start server
echo "Starting API server..."
llamafactory-cli api \
    --model_name_or_path "$MODEL_PATH" \
    --template ernie \
    --port "$PORT"
