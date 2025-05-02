#!/bin/bash
# run_semantic_experiment.sh
# Master script to run the complete semantic mapping experiment in steps

set -e  # Exit on any error

# Configuration
DPA_CSV="data/train_set.csv"
REQUIREMENTS_FILE="data/processed/requirements_symbolic.json"
MODEL_PATH="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"
TARGET_DPA="Online 1"
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

echo "========== Semantic Mapping Experiment for GDPR Compliance =========="
echo "Target DPA: ${TARGET_DPA}"
echo "=================================================================="

# Show menu of available steps
echo "Available steps:"
echo "1. Translate DPA segments to symbolic representations"
echo "2. Generate semantic rules (optimized - 1 LLM call per requirement)"
echo "3. Create LP files with semantic rules"
echo "4. Run Deolingo solver"
echo "5. Evaluate results"
echo "A. Run all steps sequentially"
echo "Q. Quit"

# Safely handle spaces in target DPA name for directory paths
DPA_DIR_NAME=$(echo ${TARGET_DPA} | tr ' ' '_')

# Ask for step selection
read -p "Enter step to run (1-5, A for all, Q to quit): " STEP

case ${STEP} in
    1)
        echo -e "\n[Step 1] Translating DPA segments to symbolic representations..."
        python translate_dpa_semantic.py \
          --dpa ${DPA_CSV} \
          --model ${MODEL_PATH} \
          --output "${OUTPUT_DIR}/dpa_symbolic.json" \
          --target "${TARGET_DPA}"
        echo "Step 1 completed. Output saved to: ${OUTPUT_DIR}/dpa_symbolic.json"
        ;;
    2)
        echo -e "\n[Step 2] Generating semantic rules (optimized approach)..."
        # Check if DPA segments file exists
        if [ ! -f "${OUTPUT_DIR}/dpa_symbolic.json" ]; then
            echo "Error: DPA segments file not found. Run Step 1 first."
            exit 1
        fi
        python generate_semantic_rules.py \
          --requirements ${REQUIREMENTS_FILE} \
          --dpa_segments "${OUTPUT_DIR}/dpa_symbolic.json" \
          --model ${MODEL_PATH} \
          --output ${OUTPUT_DIR}
        echo "Step 2 completed. Output saved to: ${OUTPUT_DIR}/semantic_rules.json"
        ;;
    3)
        echo -e "\n[Step 3] Creating LP files with semantic rules..."
        # Check if semantic rules file exists
        if [ ! -f "${OUTPUT_DIR}/semantic_rules.json" ]; then
            echo "Error: Semantic rules file not found. Run Step 2 first."
            exit 1
        fi
        python generate_lp_files_semantic.py \
          --requirements ${REQUIREMENTS_FILE} \
          --dpa_segments "${OUTPUT_DIR}/dpa_symbolic.json" \
          --semantic_rules "${OUTPUT_DIR}/semantic_rules.json" \
          --output ${OUTPUT_DIR}
        echo "Step 3 completed. LP files created in: ${OUTPUT_DIR}/dpa_${DPA_DIR_NAME}"
        ;;
    4)
        echo -e "\n[Step 4] Running Deolingo solver..."
        # Find all LP files regardless of exact directory name
        LP_FILES=$(find ${OUTPUT_DIR} -name "*.lp")
        if [ -z "${LP_FILES}" ]; then
            echo "Error: No LP files found in ${OUTPUT_DIR} directory. Run Step 3 first."
            exit 1
        fi
        
        # Create output file for results
        RESULTS_FILE="${OUTPUT_DIR}/deolingo_results.txt"
        echo "" > ${RESULTS_FILE}
        
        # Process all .lp files
        echo "Processing LP files..."
        find ${OUTPUT_DIR} -name "*.lp" | while read lp_file; do
            dpa_id=$(basename $(dirname "${lp_file}") | sed 's/dpa_//')
            req_id=$(basename "${lp_file}" .lp | sed 's/req_//')
            
            echo "Processing DPA ${dpa_id}, Requirement ${req_id}..." | tee -a ${RESULTS_FILE}
            deolingo "${lp_file}" | tee -a ${RESULTS_FILE}
            echo "--------------------------------------------------" | tee -a ${RESULTS_FILE}
        done
        echo "Step 4 completed. Results saved in: ${RESULTS_FILE}"
        ;;
    5)
        echo -e "\n[Step 5] Evaluating results..."
        # Check if deolingo results exist
        if [ ! -f "${OUTPUT_DIR}/deolingo_results.txt" ]; then
            echo "Error: Deolingo results file not found. Run Step 4 first."
            exit 1
        fi
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
        echo "Step 5 completed."
        ;;
    [aA])
        echo -e "\nRunning all steps sequentially..."
        
        echo -e "\n[Step 1] Translating DPA segments to symbolic representations..."
        python translate_dpa_semantic.py \
          --dpa ${DPA_CSV} \
          --model ${MODEL_PATH} \
          --output "${OUTPUT_DIR}/dpa_symbolic.json" \
          --target "${TARGET_DPA}"
        
        echo -e "\n[Step 2] Generating semantic rules (optimized approach)..."
        python generate_semantic_rules.py \
          --requirements ${REQUIREMENTS_FILE} \
          --dpa_segments "${OUTPUT_DIR}/dpa_symbolic.json" \
          --model ${MODEL_PATH} \
          --output ${OUTPUT_DIR}
        
        echo -e "\n[Step 3] Creating LP files with semantic rules..."
        python generate_lp_files_semantic.py \
          --requirements ${REQUIREMENTS_FILE} \
          --dpa_segments "${OUTPUT_DIR}/dpa_symbolic.json" \
          --semantic_rules "${OUTPUT_DIR}/semantic_rules.json" \
          --output ${OUTPUT_DIR}
        
        echo -e "\n[Step 4] Running Deolingo solver..."
        # Create output file for results
        RESULTS_FILE="${OUTPUT_DIR}/deolingo_results.txt"
        echo "" > ${RESULTS_FILE}
        
        # Process all .lp files
        echo "Processing LP files..."
        find ${OUTPUT_DIR} -name "*.lp" | while read lp_file; do
            dpa_id=$(basename $(dirname "${lp_file}") | sed 's/dpa_//')
            req_id=$(basename "${lp_file}" .lp | sed 's/req_//')
            
            echo "Processing DPA ${dpa_id}, Requirement ${req_id}..." | tee -a ${RESULTS_FILE}
            deolingo "${lp_file}" | tee -a ${RESULTS_FILE}
            echo "--------------------------------------------------" | tee -a ${RESULTS_FILE}
        done
        
        echo -e "\n[Step 5] Evaluating results..."
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
        echo "All steps completed successfully!"
        ;;
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