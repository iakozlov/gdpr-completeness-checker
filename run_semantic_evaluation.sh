#!/bin/bash
# run_semantic_evaluation.sh
# Script to run the complete semantic mapping evaluation pipeline

set -e  # Exit on any error

# Configuration
DPA_CSV="data/train_set.csv"
REQUIREMENTS_FILE="requirements_symbolic.json"
MODEL_PATH="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"
TARGET_DPA="Online 1"
OUTPUT_DIR="semantic_results"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.txt"

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p data/processed

# Copy requirements file to expected location if needed
if [ ! -f "data/processed/requirements_symbolic.json" ]; then
    cp ${REQUIREMENTS_FILE} data/processed/requirements_symbolic.json
fi

echo "========== DPA Completeness Checker (Semantic Mapping) =========="
echo "Using semantic mapping between requirements and DPA segments"
echo "Target DPA: ${TARGET_DPA}"
echo "=========================================================="

# Step 1: Translate DPA segments with semantic mapping
echo -e "\n[1/4] Creating semantic mappings between requirements and DPA segments..."
python translate_dpa_semantic.py \
  --dpa ${DPA_CSV} \
  --requirements data/processed/requirements_symbolic.json \
  --model ${MODEL_PATH} \
  --output ${OUTPUT_DIR} \
  --target "${TARGET_DPA}"

# Step 2: Run deolingo on all .lp files
echo -e "\n[2/4] Running Deolingo solver on the generated LP files..."
# Create a temporary script to run Deolingo
cat > ${OUTPUT_DIR}/run_deolingo_temp.sh << EOF
#!/bin/bash
# Temporary script to run deolingo on all .lp files

OUTPUT_FILE="${OUTPUT_DIR}/deolingo_results.txt"
echo "" > \$OUTPUT_FILE

# Process all .lp files
find ${OUTPUT_DIR} -name "*.lp" | while read lp_file; do
    dpa_id=\$(basename \$(dirname \$lp_file) | sed 's/dpa_//')
    req_id=\$(basename \$lp_file .lp | sed 's/req_//')
    
    echo "Processing DPA \$dpa_id, Requirement \$req_id..." | tee -a \$OUTPUT_FILE
    deolingo \$lp_file | tee -a \$OUTPUT_FILE
    echo "--------------------------------------------------" | tee -a \$OUTPUT_FILE
done

echo "All processing complete. Results saved in \$OUTPUT_FILE"
EOF

# Make the script executable and run it
chmod +x ${OUTPUT_DIR}/run_deolingo_temp.sh
${OUTPUT_DIR}/run_deolingo_temp.sh

# Step 3: Evaluate the results
echo -e "\n[3/4] Evaluating Deolingo results..."
python evaluate_results.py \
  --results "${OUTPUT_DIR}/deolingo_results.txt" \
  --dpa ${DPA_CSV} \
  --output ${EVALUATION_OUTPUT} \
  --target "${TARGET_DPA}"

# Step 4: Display summary
echo -e "\n[4/4] Evaluation complete!"
echo "=========================================================="
echo "Results Summary:"
grep -A 10 "Evaluation Results for DPA" ${EVALUATION_OUTPUT}
echo -e "\nDetailed results saved to: ${EVALUATION_OUTPUT}"
echo "=========================================================="

# Compare with previous approach if available
if [ -f "results_v2/evaluation_results.txt" ]; then
    echo -e "\n========== Comparison with Previous Approach =========="
    echo "Previous approach (vocabulary alignment):"
    grep -A 3 "Predicted:" results_v2/evaluation_results.txt
    echo "Current approach (semantic mapping):"
    grep -A 3 "Predicted:" ${EVALUATION_OUTPUT}
    echo "=========================================================="
fi