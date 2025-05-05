#!/bin/bash
# run_semantic_experiment.sh
# Master script to run the complete semantic mapping experiment with deontic logic

set -e  # Exit on any error

# Configuration
DPA_CSV="data/train_set.csv"
REQUIREMENTS_FILE="data/processed/requirements_symbolic.json"
MODEL_PATH="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"
TARGET_DPA="Online 1"
MAX_SEGMENTS=10  # Limit to first 10 segments for speed
OUTPUT_DIR="semantic_results"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.txt"

# Create directory structure
mkdir -p ${OUTPUT_DIR}

# Copy requirements file if needed
if [ ! -f "${REQUIREMENTS_FILE}" ] && [ -f "requirements_symbolic.json" ]; then
    mkdir -p data/processed
    cp requirements_symbolic.json ${REQUIREMENTS_FILE}
    echo "Copied requirements file to expected location"
fi

echo "========== Semantic Mapping Experiment with Deontic Logic =========="
echo "Target DPA: ${TARGET_DPA} (first ${MAX_SEGMENTS} segments)"
echo "=================================================================="

# Show menu of available steps
echo "Available steps:"
echo "1. Translate DPA segments to deontic logic"
echo "2. Generate semantic rules between requirements and DPA"
echo "3. Run Deolingo solver"
echo "4. Evaluate results"
echo "A. Run all steps sequentially"
echo "Q. Quit"

# Safely handle spaces in target DPA name for directory paths
DPA_DIR_NAME=$(echo ${TARGET_DPA} | tr ' ' '_')

# Function to run step 1
run_step_1() {
    echo -e "\n[Step 1] Translating DPA segments to deontic logic..."
    python translate_dpa_semantic.py \
      --dpa ${DPA_CSV} \
      --model ${MODEL_PATH} \
      --output "${OUTPUT_DIR}/dpa_deontic.json" \
      --target "${TARGET_DPA}" \
      --max_segments ${MAX_SEGMENTS}
    echo "Step 1 completed. Output saved to: ${OUTPUT_DIR}/dpa_deontic.json"
}

# Function to run step 2
run_step_2() {
    echo -e "\n[Step 2] Generating semantic rules and LP files..."
    # Check if DPA deontic file exists
    if [ ! -f "${OUTPUT_DIR}/dpa_deontic.json" ]; then
        echo "Error: DPA deontic file not found. Run Step 1 first."
        return 1
    fi
    python generate_semantic_rules.py \
      --requirements ${REQUIREMENTS_FILE} \
      --dpa_segments "${OUTPUT_DIR}/dpa_deontic.json" \
      --model ${MODEL_PATH} \
      --output ${OUTPUT_DIR}
    echo "Step 2 completed. LP files created in: ${OUTPUT_DIR}/dpa_${DPA_DIR_NAME}"
}

# Function to run step 3
run_step_3() {
    echo -e "\n[Step 3] Running Deolingo solver..."
    # Check if LP files exist
    if [ ! -d "${OUTPUT_DIR}/dpa_${DPA_DIR_NAME}" ]; then
        echo "Error: LP files directory not found. Run Step 2 first."
        return 1
    fi
    
    # Create output file for results
    RESULTS_FILE="${OUTPUT_DIR}/deolingo_results.txt"
    echo "" > ${RESULTS_FILE}
    
    # Process all .lp files
    echo "Processing LP files..."
    find "${OUTPUT_DIR}/dpa_${DPA_DIR_NAME}" -name "*.lp" | while read lp_file; do
        req_id=$(basename "${lp_file}" .lp | sed 's/req_//')
        
        echo "Processing DPA ${DPA_DIR_NAME}, Requirement ${req_id}..." | tee -a ${RESULTS_FILE}
        deolingo "${lp_file}" | tee -a ${RESULTS_FILE}
        echo "--------------------------------------------------" | tee -a ${RESULTS_FILE}
    done
    echo "Step 3 completed. Results saved in: ${RESULTS_FILE}"
}

# Function to run step 4
run_step_4() {
    echo -e "\n[Step 4] Evaluating results..."
    # Check if deolingo results exist
    if [ ! -f "${OUTPUT_DIR}/deolingo_results.txt" ]; then
        echo "Error: Deolingo results file not found. Run Step 3 first."
        return 1
    fi
    python evaluate_semantic_results.py \
      --results "${OUTPUT_DIR}/deolingo_results.txt" \
      --dpa ${DPA_CSV} \
      --requirements_text "data/requirements/ground_truth_requirements.txt" \
      --output ${EVALUATION_OUTPUT} \
      --target "${TARGET_DPA}"
      
    # Display summary
    echo "=================================================="
    echo "Results Summary:"
    if [ -f ${EVALUATION_OUTPUT} ]; then
        head -15 ${EVALUATION_OUTPUT}
    else
        echo "Evaluation results file not found"
    fi
    echo "=================================================="
    
    # Compare with previous approach if available
    if [ -f "results_v2/evaluation_results.txt" ]; then
        echo -e "\n========== Comparison with Previous Approach =========="
        echo "Previous approach (vocabulary alignment):"
        grep -A 3 "Predicted:" results_v2/evaluation_results.txt
        echo "Current approach (semantic mapping with deontic logic):"
        grep -A 3 "Predicted:" ${EVALUATION_OUTPUT}
        echo "=================================================="
    fi
    echo "Step 4 completed."
}

# Function to run all steps
run_all_steps() {
    run_step_1 && run_step_2 && run_step_3 && run_step_4
}

# Ask for step selection
read -p "Enter step to run (1-4, A for all, Q to quit): " STEP

case ${STEP} in
    1) run_step_1 ;;
    2) run_step_2 ;;
    3) run_step_3 ;;
    4) run_step_4 ;;
    [aA]) run_all_steps ;;
    [qQ])
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

echo "Done!"