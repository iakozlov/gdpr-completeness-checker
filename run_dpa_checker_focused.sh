#!/bin/bash
# run_dpa_checker_focused.sh - Run focused DPA compliance checking

set -e  # Exit on any error

# Configuration
REQUIREMENTS_FILE="data/processed/requirements_symbolic.json"
DPA_SEGMENTS_FILE="data/processed/dpa_segments.json"
MODEL_PATH="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"
OUTPUT_DIR="results/focused"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.json"

# Display header
echo "========== DPA Compliance Checker (Focused) =========="
echo "Target requirement: #5 (Article 32 security measures)"
echo "DPA segments: First 30"
echo "======================================================="

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Step 1: Generate focused LP files
echo -e "\n[Step 1] Generating focused LP files"
python generate_focused_lp_files.py \
  --requirements ${REQUIREMENTS_FILE} \
  --dpa_segments ${DPA_SEGMENTS_FILE} \
  --model ${MODEL_PATH} \
  --output ${OUTPUT_DIR}

# Find the LP directory
LP_DIR="${OUTPUT_DIR}/req_5"
if [ ! -d "${LP_DIR}" ]; then
  echo "Error: LP directory not found: ${LP_DIR}"
  exit 1
fi

# Step 2: Run deolingo evaluation
echo -e "\n[Step 2] Running deolingo evaluation"
python evaluate_individual_results.py \
  --lp_dir ${LP_DIR} \
  --output ${EVALUATION_OUTPUT}

# Check if evaluation results exist
if [ ! -f "${EVALUATION_OUTPUT}" ]; then
  echo "Error: Evaluation results not found: ${EVALUATION_OUTPUT}"
  exit 1
fi

# Step 3: Display summary
echo -e "\n[Step 3] Results Summary"
python -c "import json; r=json.load(open('${EVALUATION_OUTPUT}')); print(f'Total segments: {r[\"total_segments\"]}\nSatisfies: {r[\"summary\"][\"satisfies\"]}\nViolates: {r[\"summary\"][\"violates\"]}\nNot mentioned: {r[\"summary\"][\"not_mentioned\"]}\nErrors: {r[\"summary\"][\"error\"]}')"

# Display completion message
echo -e "\nDPA compliance checking complete!"
echo "Detailed results saved to: ${EVALUATION_OUTPUT}"