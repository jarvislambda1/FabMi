#!/bin/bash
# FabMi: Evaluate fine-tuned model on test set
# Usage: bash scripts/training/evaluate.sh [--workers 4]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "FabMi: Running Evaluation"
echo "=========================================="
echo "Test set: $PROJECT_ROOT/data/splits/test.json"
echo "Model: $PROJECT_ROOT/models/ernie_semi_rca_merged"
echo ""

cd "$PROJECT_ROOT"

python eval/eval_finetuned_parallel.py \
    --model "$PROJECT_ROOT/models/ernie_semi_rca_merged" \
    "$@"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results: $PROJECT_ROOT/eval/results/"
echo "=========================================="
