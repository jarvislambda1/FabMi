#!/bin/bash
# FabMi: Merge LoRA adapter with base model
# Usage: bash scripts/training/merge_lora.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "FabMi: Merging LoRA with Base Model"
echo "=========================================="

llamafactory-cli export "$PROJECT_ROOT/configs/merge_config.yaml"

echo ""
echo "=========================================="
echo "Merge Complete!"
echo "=========================================="
echo "Merged model: $PROJECT_ROOT/models/ernie_semi_rca_merged"
echo "Next: bash scripts/serving/serve_model.sh"
echo "=========================================="
