#!/bin/bash
# run_evaluation_v2.sh
# Script to run the complete evaluation pipeline with the improved vocabulary-aware translation

set -e  # Exit on any error

# Configuration
DPA_CSV="data/train_set.csv"
REQUIREMENTS_FILE="requirements_symbolic.json"
MODEL_PATH="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"
TARGET_DPA="Online 1"
OUTPUT_DIR="results_v2"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.txt"

# Create output directory
mkdir -p ${OUTPUT_DIR}
mkdir -p data/processed

echo "========== DPA Completeness Checker V2 =========="
echo "Using vocabulary-aligned translation approach"
echo "Target DPA: ${TARGET_DPA}"
echo "================================================="

# Step 2: Generate LP files from symbolic representations
echo -e "\n[2/5] Generating logic program (.lp) files..."
python generate_lp_files.py \
  --requirements data/processed/requirements_symbolic.json \
  --dpa "data/processed/dpa_segments_symbolic_v2.json" \
  --output ${OUTPUT_DIR}

# Step 3: Run deolingo on all .lp files
echo -e "\n[3/5] Running Deolingo solver on logic programs..."
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

# Step 4: Evaluate the results
echo -e "\n[4/5] Evaluating Deolingo results..."
python evaluate_results.py \
  --results "${OUTPUT_DIR}/deolingo_results.txt" \
  --dpa ${DPA_CSV} \
  --output ${EVALUATION_OUTPUT} \
  --target "${TARGET_DPA}"

# Step 5: Display summary
echo -e "\n[5/5] Evaluation complete!"
echo "================================================="
echo "Results Summary:"
grep -A 4 "Evaluation Results for DPA" ${EVALUATION_OUTPUT}
echo -e "\nDetailed results saved to: ${EVALUATION_OUTPUT}"
echo "================================================="