#!/bin/bash
# evaluate_semantic_results.sh
# Script to evaluate the results of the semantic mapping approach

set -e  # Exit on any error

# Configuration
DPA_CSV="data/train_set.csv"
OUTPUT_DIR="semantic_results"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.txt"
TARGET_DPA="Online 1"  # Use space format for CSV lookup

echo "========== Evaluating Semantic Mapping Results =========="
echo "Target DPA: ${TARGET_DPA}"
echo "========================================================"

# Check if deolingo results exist
if [ ! -f "${OUTPUT_DIR}/deolingo_results.txt" ]; then
    echo "Error: Deolingo results file not found: ${OUTPUT_DIR}/deolingo_results.txt"
    echo "Please run run_deolingo_semantic.sh first"
    exit 1
fi

# Run evaluation
echo "Evaluating results..."
python evaluate_results.py \
  --results "${OUTPUT_DIR}/deolingo_results.txt" \
  --dpa ${DPA_CSV} \
  --output ${EVALUATION_OUTPUT} \
  --target "${TARGET_DPA}"

# Display summary
echo "=================================================="
echo "Results Summary:"
grep -A 10 "Evaluation Results for DPA" ${EVALUATION_OUTPUT}
echo -e "\nDetailed results saved to: ${EVALUATION_OUTPUT}"
echo "=================================================="

# Compare with previous approach if available
if [ -f "results_v2/evaluation_results.txt" ]; then
    echo -e "\n========== Comparison with Previous Approach =========="
    echo "Previous approach (vocabulary alignment):"
    grep -A 3 "Predicted:" results_v2/evaluation_results.txt
    echo "Current approach (semantic mapping):"
    grep -A 3 "Predicted:" ${EVALUATION_OUTPUT}
    
    # Compare requirements satisfaction
    echo -e "\nRequirements satisfaction comparison:"
    echo "Previous approach - satisfied requirements:"
    grep -A 2 "Ground Truth Covered Requirements:" results_v2/evaluation_results.txt | tail -n 1
    echo "Semantic approach - satisfied requirements:"
    grep -A 2 "Ground Truth Covered Requirements:" ${EVALUATION_OUTPUT} | tail -n 1
    echo "=================================================="
fi