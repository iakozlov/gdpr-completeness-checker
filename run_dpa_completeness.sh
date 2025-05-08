#!/bin/bash
# run_dpa_completeness.sh
# Master script to run the DPA completeness evaluation pipeline
# Evaluates completeness of Online 1 DPA against requirements R7-R25

set -e  # Exit on any error

# Configuration
DPA_CSV="data/train_set.csv"
REQUIREMENTS_FILE="data/requirements/ground_truth_requirements.txt"
MODEL_PATH="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"
OUTPUT_DIR="results"
DEOLINGO_RESULTS="${OUTPUT_DIR}/deolingo_results.txt"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.json"
REQUIREMENTS_DEONTIC="${OUTPUT_DIR}/requirements_deontic.json"
TARGET_DPA="Online 1"
REQ_IDS="all"  # Focus on requirement 6 by default
MAX_SEGMENTS=30  # Limit to 30 segments by default

# Command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --req_ids)
      REQ_IDS="$2"
      shift 2
      ;;
    --max_segments)
      MAX_SEGMENTS="$2"
      shift 2
      ;;
    --target_dpa)
      TARGET_DPA="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "========== DPA Completeness Checker =========="
echo "Using Deontic Logic and Answer Set Programming"
echo "Evaluating DPA '${TARGET_DPA}'"
echo "Focus on requirement(s): ${REQ_IDS}"
echo "Using ${MAX_SEGMENTS} segments"
echo "=============================================="

# Show menu of available steps
echo "Available steps:"
echo "1. Translate all requirements to deontic logic"
echo "2. Generate LP files for specified requirements and segments"
echo "3. Run Deolingo solver"
echo "4. Evaluate DPA completeness"
echo "A. Run all steps sequentially"
echo "Q. Quit"

read -p "Enter step to run (1-4, A for all, Q to quit): " STEP

# Function to run deolingo with error handling
run_deolingo() {
    local lp_file=$1
    local req_id=$2
    local segment_id=$3
    
    # Run deolingo and capture output
    deolingo_output=$(deolingo ${lp_file} 2>&1) || true
    
    # Check if there was an error
    if [[ $deolingo_output == *"ERROR"* || $deolingo_output == *"error"* ]]; then
        echo "Processing DPA ${TARGET_DPA}, Requirement ${req_id}, Segment ${segment_id}..." >> ${DEOLINGO_RESULTS}
        echo "Error processing file: ${lp_file}" >> ${DEOLINGO_RESULTS}
        echo "Answer: not_mentioned" >> ${DEOLINGO_RESULTS}
        echo "--------------------------------------------------" >> ${DEOLINGO_RESULTS}
        echo "Warning: Error in file ${lp_file}. Skipping and continuing..." >&2
    else
        echo "Processing DPA ${TARGET_DPA}, Requirement ${req_id}, Segment ${segment_id}..." >> ${DEOLINGO_RESULTS}
        echo "${deolingo_output}" >> ${DEOLINGO_RESULTS}
        echo "--------------------------------------------------" >> ${DEOLINGO_RESULTS}
    fi
}

case ${STEP} in
    1)
        echo -e "\n[Step 1] Translating all requirements to deontic logic..."
        python translate_requirements.py \
          --requirements ${REQUIREMENTS_FILE} \
          --model ${MODEL_PATH} \
          --output "${REQUIREMENTS_DEONTIC}"
        echo "Step 1 completed. Output saved to: ${REQUIREMENTS_DEONTIC}"
        ;;
    2)
        echo -e "\n[Step 2] Generating LP files for requirement(s) ${REQ_IDS} and ${MAX_SEGMENTS} segments..."
        python generate_lp_files.py \
          --requirements "${REQUIREMENTS_DEONTIC}" \
          --dpa ${DPA_CSV} \
          --model ${MODEL_PATH} \
          --output "${OUTPUT_DIR}/lp_files" \
          --target_dpa "${TARGET_DPA}" \
          --max_segments ${MAX_SEGMENTS}
          #--req_ids "${REQ_IDS}" \
        echo "Step 2 completed. LP files generated in: ${OUTPUT_DIR}/lp_files"
        ;;
    3)
        echo -e "\n[Step 3] Running Deolingo solver..."
        DPA_DIR="${OUTPUT_DIR}/lp_files/dpa_${TARGET_DPA//' '/_}"
        
        if [ ! -d "${DPA_DIR}" ]; then
            echo "Error: LP files directory not found at ${DPA_DIR}. Run Step 2 first."
            exit 1
        fi
        
        # Create output file for results
        echo "" > ${DEOLINGO_RESULTS}
        
        # Process requirements - use only those specified by REQ_IDS if not "all"
        if [ "${REQ_IDS}" == "all" ]; then
            REQ_DIRS=$(find "${DPA_DIR}" -type d -name "req_*")
        else
            REQ_DIRS=""
            for REQ_ID in $(echo ${REQ_IDS} | tr ',' ' '); do
                if [ -d "${DPA_DIR}/req_${REQ_ID}" ]; then
                    REQ_DIRS="${REQ_DIRS} ${DPA_DIR}/req_${REQ_ID}"
                fi
            done
        fi
        
        for req_dir in ${REQ_DIRS}; do
            req_id=$(basename ${req_dir} | sed 's/req_//')
            echo "Processing requirement ${req_id}..."
            
            # Process all .lp files for this requirement
            find "${req_dir}" -name "*.lp" | while read lp_file; do
                # Extract segment ID from the file path
                segment_id=$(basename $lp_file | sed 's/segment_//' | sed 's/\.lp//')
                
                # Run deolingo with error handling
                run_deolingo "${lp_file}" "${req_id}" "${segment_id}"
            done
        done
        echo "Step 3 completed. Results saved in: ${DEOLINGO_RESULTS}"
        ;;
    4)
        echo -e "\n[Step 4] Evaluating DPA completeness..."
        if [ ! -f "${DEOLINGO_RESULTS}" ]; then
            echo "Error: Deolingo results file not found. Run Step 3 first."
            exit 1
        fi
        python evaluate_completeness.py \
          --results ${DEOLINGO_RESULTS} \
          --dpa ${DPA_CSV} \
          --output ${EVALUATION_OUTPUT} \
          --target_dpa "${TARGET_DPA}" \
          --max_segments "${MAX_SEGMENTS}"
        
        echo "Step 4 completed. Evaluation results saved to: ${EVALUATION_OUTPUT}"
        ;;
    [aA])
        echo -e "\nRunning all steps sequentially..."
        
        echo -e "\n[Step 1] Translating all requirements to deontic logic..."
        python translate_requirements.py \
          --requirements ${REQUIREMENTS_FILE} \
          --model ${MODEL_PATH} \
          --output "${REQUIREMENTS_DEONTIC}"
        
        echo -e "\n[Step 2] Generating LP files for requirement(s) ${REQ_IDS} and ${MAX_SEGMENTS} segments..."
        python generate_lp_files.py \
          --requirements "${REQUIREMENTS_DEONTIC}" \
          --dpa ${DPA_CSV} \
          --model ${MODEL_PATH} \
          --output "${OUTPUT_DIR}/lp_files" \
          --target_dpa "${TARGET_DPA}" \
          --max_segments ${MAX_SEGMENTS}
          #--req_ids "${REQ_IDS}" \
        
        echo -e "\n[Step 3] Running Deolingo solver..."
        DPA_DIR="${OUTPUT_DIR}/lp_files/dpa_${TARGET_DPA//' '/_}"
        
        # Create output file for results
        echo "" > ${DEOLINGO_RESULTS}
        
        # Process requirements - use only those specified by REQ_IDS
        if [ "${REQ_IDS}" == "all" ]; then
            REQ_DIRS=$(find "${DPA_DIR}" -type d -name "req_*")
        else
            REQ_DIRS=""
            for REQ_ID in $(echo ${REQ_IDS} | tr ',' ' '); do
                if [ -d "${DPA_DIR}/req_${REQ_ID}" ]; then
                    REQ_DIRS="${REQ_DIRS} ${DPA_DIR}/req_${REQ_ID}"
                fi
            done
        fi
        
        for req_dir in ${REQ_DIRS}; do
            req_id=$(basename ${req_dir} | sed 's/req_//')
            echo "Processing requirement ${req_id}..."
            
            # Process all .lp files for this requirement
            find "${req_dir}" -name "*.lp" | while read lp_file; do
                # Extract segment ID from the file path
                segment_id=$(basename $lp_file | sed 's/segment_//' | sed 's/\.lp//')
                
                # Run deolingo with error handling
                run_deolingo "${lp_file}" "${req_id}" "${segment_id}"
            done
        done
        
        echo -e "\n[Step 4] Evaluating DPA completeness..."
        python evaluate_completeness.py \
          --results ${DEOLINGO_RESULTS} \
          --dpa ${DPA_CSV} \
          --output ${EVALUATION_OUTPUT} \
          --target_dpa "${TARGET_DPA}" \
          --max_segments "${MAX_SEGMENTS}"
        
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