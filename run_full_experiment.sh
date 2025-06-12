#!/bin/bash
# Run baseline and symbolic pipelines and produce a comparison

set -e

BASELINE_DIR="results/experiment/baseline"
SYMBOLIC_DIR="results/experiment/symbolic"
MODEL="gemma3:27b"
DPA_FILE="data/test_set.csv"
TARGET_DPAS="Online 124,Online 132"
REQ_IDS="all"
MAX_SEGMENTS=0

# Run baseline pipeline
bash run_baseline_evaluation.sh \
  --step all \
  --ollama_model "$MODEL" \
  --dpa_file "$DPA_FILE" \
  --target_dpas "$TARGET_DPAS" \
  --req_ids "$REQ_IDS" \
  --max_segments "$MAX_SEGMENTS" \
  --output_dir "$BASELINE_DIR"

# Run symbolic pipeline (all steps)
echo "A" | bash run_dpa_completeness_ollama.sh \
  --model "$MODEL" \
  --output_dir "$SYMBOLIC_DIR" \
  --target_dpas "$TARGET_DPAS" \
  --req_ids "$REQ_IDS" \
  --max_segments "$MAX_SEGMENTS" > /dev/null

# Compare results
python compare_experiments.py \
  --baseline_dir "$BASELINE_DIR" \
  --deolingo_results "$SYMBOLIC_DIR/deolingo_results.txt" \
  --lp_root "$SYMBOLIC_DIR/lp_files" \
  --output "$SYMBOLIC_DIR/comparison.json"

echo "Comparison saved to $SYMBOLIC_DIR/comparison.json"
