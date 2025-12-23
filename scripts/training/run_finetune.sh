#!/bin/bash
# FabMi: Fine-tuning ERNIE 4.5 0.3B on Semiconductor RCA data
# Usage: bash scripts/training/run_finetune.sh [config]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG=${1:-"$PROJECT_ROOT/configs/ernie_semi_rca_10ep.yaml"}

echo "=========================================="
echo "FabMi: Starting ERNIE Fine-tuning"
echo "=========================================="
echo "Config: $CONFIG"
echo "Dataset: $PROJECT_ROOT/data/splits/train.json"
echo ""

# Run training with LLaMA-Factory
llamafactory-cli train "$CONFIG"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Next steps:"
echo "1. Merge LoRA: bash scripts/training/merge_lora.sh"
echo "2. Serve model: bash scripts/serving/serve_model.sh"
echo "=========================================="
