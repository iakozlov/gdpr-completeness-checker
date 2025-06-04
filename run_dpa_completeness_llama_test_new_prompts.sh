#!/bin/bash
# run_dpa_completeness_llama_test_new_prompts.sh
# Test script for the new requirement-specific prompts approach
# Tests on DPA "Online 124" with 50 segments and requirements 1-6

set -e  # Exit on any error

# Configuration for testing new approach
DPA_CSV="data/test_set.csv"
REQUIREMENTS_FILE="data/requirements/ground_truth_requirements.txt"
MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"  # Llama 3.3-70B model via Together.ai API
OUTPUT_DIR="results/llama_experiment/new_prompts_test"
DEOLINGO_RESULTS="${OUTPUT_DIR}/deolingo_results.txt"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.json"
PARAGRAPH_OUTPUT="${OUTPUT_DIR}/paragraph_metrics.json"
REQUIREMENTS_DEONTIC="data/requirements/requirements_deontic_ai_generated.json"

# Test-specific configuration
TARGET_DPAS=("Online 124")  # Specific DPA for testing
REQ_IDS="1,2,3,4,5,6"  # Focus on requirements 1-6
MAX_SEGMENTS=50  # Limit to 50 segments for testing
REQUIREMENT_PROMPTS="requirement_prompts.json"  # New requirement-specific prompts

# Command line arguments (with defaults for testing)
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
    --target_dpas)
      IFS=',' read -ra TARGET_DPAS <<< "$2"
      shift 2
      ;;
    --requirement_prompts)
      REQUIREMENT_PROMPTS="$2"
      shift 2
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "========== DPA Completeness Checker - New Prompts Test =========="
echo "Using Llama 3.3-70B Model via Together.ai API"
echo "Testing NEW REQUIREMENT-SPECIFIC PROMPTS approach"
echo "Using Deontic Logic and Answer Set Programming"
echo "Evaluating DPAs: ${TARGET_DPAS[*]}"
echo "Focus on requirement(s): ${REQ_IDS}"
echo "Using ${MAX_SEGMENTS} segments for testing"
echo "Requirement prompts file: ${REQUIREMENT_PROMPTS}"
echo "=============================================="

# Check if requirement prompts file exists
if [ ! -f "${REQUIREMENT_PROMPTS}" ]; then
    echo "Error: Requirement prompts file ${REQUIREMENT_PROMPTS} not found!"
    echo "Please run: python generate_requirement_prompts.py first"
    exit 1
fi

# Show menu of available steps
echo "Available steps:"
echo "1. Translate all requirements to deontic logic (if needed)"
echo "2. Generate LP files using NEW requirement-specific prompts"
echo "3. Run Deolingo solver for all DPAs"
echo "4. Evaluate DPA completeness (aggregated results)"
echo "5. Calculate paragraph-level metrics (aggregated results)"
echo "A. Run all steps sequentially"
echo "Q. Quit"

read -p "Enter step to run (1-5, A for all, Q to quit): " STEP

# Function to run deolingo with error handling
run_deolingo() {
    local lp_file=$1
    local req_id=$2
    local segment_id=$3
    local dpa_name=$4
    
    # Run deolingo and capture output
    deolingo_output=$(deolingo ${lp_file} 2>&1) || true
    
    # Check if there was an error
    if [[ $deolingo_output == *"ERROR"* || $deolingo_output == *"error"* ]]; then
        echo "Processing DPA ${dpa_name}, Requirement ${req_id}, Segment ${segment_id}..." >> ${DEOLINGO_RESULTS}
        echo "Error processing file: ${lp_file}" >> ${DEOLINGO_RESULTS}
        echo "Answer: not_mentioned" >> ${DEOLINGO_RESULTS}
        echo "--------------------------------------------------" >> ${DEOLINGO_RESULTS}
        echo "Warning: Error in file ${lp_file}. Skipping and continuing..." >&2
    else
        echo "Processing DPA ${dpa_name}, Requirement ${req_id}, Segment ${segment_id}..." >> ${DEOLINGO_RESULTS}
        echo "${deolingo_output}" >> ${DEOLINGO_RESULTS}
        echo "--------------------------------------------------" >> ${DEOLINGO_RESULTS}
    fi
}

case ${STEP} in
    1)
        echo -e "\n[Step 1] Translating all requirements to deontic logic (if needed)..."
        if [ ! -f "${REQUIREMENTS_DEONTIC}" ]; then
            python translate_requirements.py \
              --requirements ${REQUIREMENTS_FILE} \
              --model ${MODEL} \
              --output "${REQUIREMENTS_DEONTIC}"
            echo "Step 1 completed. Output saved to: ${REQUIREMENTS_DEONTIC}"
        else
            echo "Requirements already translated. Using existing file: ${REQUIREMENTS_DEONTIC}"
        fi
        ;;
    2)
        echo -e "\n[Step 2] Generating LP files using NEW requirement-specific prompts..."
        echo "Testing requirements: ${REQ_IDS}"
        echo "Testing ${MAX_SEGMENTS} segments from DPA: ${TARGET_DPAS[*]}"
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            echo "Processing DPA: ${TARGET_DPA}"
            python generate_lp_files.py \
              --requirements "${REQUIREMENTS_DEONTIC}" \
              --dpa ${DPA_CSV} \
              --model ${MODEL} \
              --output "${OUTPUT_DIR}/lp_files" \
              --target_dpa "${TARGET_DPA}" \
              --max_segments ${MAX_SEGMENTS} \
              --req_ids "${REQ_IDS}" \
              --requirement_prompts "${REQUIREMENT_PROMPTS}"
        done
        echo "Step 2 completed. LP files with NEW prompts generated in: ${OUTPUT_DIR}/lp_files"
        ;;
    3)
        echo -e "\n[Step 3] Running Deolingo solver for all DPAs..."
        # Create output file for results
        echo "" > ${DEOLINGO_RESULTS}
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            DPA_DIR="${OUTPUT_DIR}/lp_files/dpa_${TARGET_DPA//' '/_}"
            
            if [ ! -d "${DPA_DIR}" ]; then
                echo "Error: LP files directory not found at ${DPA_DIR} for DPA ${TARGET_DPA}. Run Step 2 first."
                continue
            fi
            
            echo "Processing DPA: ${TARGET_DPA}"
            
            # Process only the specified requirements (1-6)
            REQ_DIRS=""
            for REQ_ID in $(echo ${REQ_IDS} | tr ',' ' '); do
                if [ -d "${DPA_DIR}/req_${REQ_ID}" ]; then
                    REQ_DIRS="${REQ_DIRS} ${DPA_DIR}/req_${REQ_ID}"
                fi
            done
            
            for req_dir in ${REQ_DIRS}; do
                req_id=$(basename ${req_dir} | sed 's/req_//')
                echo "  Processing requirement ${req_id}..."
                
                # Process all .lp files for this requirement
                find "${req_dir}" -name "*.lp" | while read lp_file; do
                    # Extract segment ID from the file path
                    segment_id=$(basename $lp_file | sed 's/segment_//' | sed 's/\.lp//')
                    
                    # Run deolingo with error handling
                    run_deolingo "${lp_file}" "${req_id}" "${segment_id}" "${TARGET_DPA}"
                done
            done
        done
        echo "Step 3 completed. Results saved in: ${DEOLINGO_RESULTS}"
        ;;
    4)
        echo -e "\n[Step 4] Evaluating DPA completeness (aggregated results)..."
        if [ ! -f "${DEOLINGO_RESULTS}" ]; then
            echo "Error: Deolingo results file not found. Run Step 3 first."
            exit 1
        fi
        
        # Process each DPA and then aggregate results
        TEMP_OUTPUTS=""
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            TEMP_OUTPUT="${OUTPUT_DIR}/evaluation_${TARGET_DPA//' '/_}.json"
            if [ -z "$TEMP_OUTPUTS" ]; then
                TEMP_OUTPUTS="${TEMP_OUTPUT}"
            else
                TEMP_OUTPUTS="${TEMP_OUTPUTS},${TEMP_OUTPUT}"
            fi
            
            echo "Processing DPA: ${TARGET_DPA}"
            python evaluate_completeness.py \
              --results ${DEOLINGO_RESULTS} \
              --dpa ${DPA_CSV} \
              --output ${TEMP_OUTPUT} \
              --target_dpa "${TARGET_DPA}" \
              --max_segments "${MAX_SEGMENTS}"
        done
        
        # Aggregate results
        python aggregate_evaluations.py \
          --input_files "${TEMP_OUTPUTS}" \
          --output ${EVALUATION_OUTPUT}
        
        echo "Step 4 completed. Aggregated evaluation results saved to: ${EVALUATION_OUTPUT}"
        ;;
    5)
        echo -e "\n[Step 5] Calculating paragraph-level metrics (aggregated results)..."
        if [ ! -f "${DEOLINGO_RESULTS}" ]; then
            echo "Error: Deolingo results file not found. Run Step 3 first."
            exit 1
        fi
        if [ ! -f "${EVALUATION_OUTPUT}" ]; then
            echo "Error: Evaluation results file not found. Run Step 4 first."
            exit 1
        fi
        
        # Process each DPA and then aggregate results
        TEMP_OUTPUTS=""
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            TEMP_OUTPUT="${OUTPUT_DIR}/paragraph_${TARGET_DPA//' '/_}.json"
            if [ -z "$TEMP_OUTPUTS" ]; then
                TEMP_OUTPUTS="${TEMP_OUTPUT}"
            else
                TEMP_OUTPUTS="${TEMP_OUTPUTS},${TEMP_OUTPUT}"
            fi
            
            echo "Processing DPA: ${TARGET_DPA}"
            python paragraph_metrics.py \
              --results ${DEOLINGO_RESULTS} \
              --dpa ${DPA_CSV} \
              --evaluation ${EVALUATION_OUTPUT} \
              --output ${TEMP_OUTPUT} \
              --target_dpa "${TARGET_DPA}" \
              --max_segments "${MAX_SEGMENTS}"
        done
        
        # Aggregate results
        python aggregate_paragraph_metrics.py \
          --input_files "${TEMP_OUTPUTS}" \
          --output ${PARAGRAPH_OUTPUT}
        
        echo "Step 5 completed. Paragraph metrics saved to: ${PARAGRAPH_OUTPUT}"
        ;;
    A|a)
        echo -e "\nRunning all steps sequentially for NEW prompts testing..."
        
        # Step 1 (conditional)
        echo -e "\n[Step 1] Checking/translating requirements to deontic logic..."
        if [ ! -f "${REQUIREMENTS_DEONTIC}" ]; then
            python translate_requirements.py \
              --requirements ${REQUIREMENTS_FILE} \
              --model ${MODEL} \
              --output "${REQUIREMENTS_DEONTIC}"
        else
            echo "Requirements already translated. Using existing file: ${REQUIREMENTS_DEONTIC}"
        fi
        
        # Step 2 - NEW PROMPTS
        echo -e "\n[Step 2] Generating LP files with NEW requirement-specific prompts..."
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            echo "Processing DPA: ${TARGET_DPA}"
            python generate_lp_files.py \
              --requirements "${REQUIREMENTS_DEONTIC}" \
              --dpa ${DPA_CSV} \
              --model ${MODEL} \
              --output "${OUTPUT_DIR}/lp_files" \
              --target_dpa "${TARGET_DPA}" \
              --max_segments ${MAX_SEGMENTS} \
              --req_ids "${REQ_IDS}" \
              --requirement_prompts "${REQUIREMENT_PROMPTS}"
        done
        
        # Step 3
        echo -e "\n[Step 3] Running Deolingo solver..."
        echo "" > ${DEOLINGO_RESULTS}
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            DPA_DIR="${OUTPUT_DIR}/lp_files/dpa_${TARGET_DPA//' '/_}"
            if [ ! -d "${DPA_DIR}" ]; then
                echo "Error: LP files directory not found at ${DPA_DIR} for DPA ${TARGET_DPA}. Skipping..."
                continue
            fi
            
            echo "Processing DPA: ${TARGET_DPA}"
            REQ_DIRS=""
            for REQ_ID in $(echo ${REQ_IDS} | tr ',' ' '); do
                if [ -d "${DPA_DIR}/req_${REQ_ID}" ]; then
                    REQ_DIRS="${REQ_DIRS} ${DPA_DIR}/req_${REQ_ID}"
                fi
            done
            
            for req_dir in ${REQ_DIRS}; do
                req_id=$(basename ${req_dir} | sed 's/req_//')
                echo "  Processing requirement ${req_id}..."
                find "${req_dir}" -name "*.lp" | while read lp_file; do
                    segment_id=$(basename $lp_file | sed 's/segment_//' | sed 's/\.lp//')
                    run_deolingo "${lp_file}" "${req_id}" "${segment_id}" "${TARGET_DPA}"
                done
            done
        done
        
        # Step 4
        echo -e "\n[Step 4] Evaluating DPA completeness..."
        TEMP_OUTPUTS=""
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            TEMP_OUTPUT="${OUTPUT_DIR}/evaluation_${TARGET_DPA//' '/_}.json"
            if [ -z "$TEMP_OUTPUTS" ]; then
                TEMP_OUTPUTS="${TEMP_OUTPUT}"
            else
                TEMP_OUTPUTS="${TEMP_OUTPUTS},${TEMP_OUTPUT}"
            fi
            
            echo "Processing DPA: ${TARGET_DPA}"
            python evaluate_completeness.py \
              --results ${DEOLINGO_RESULTS} \
              --dpa ${DPA_CSV} \
              --output ${TEMP_OUTPUT} \
              --target_dpa "${TARGET_DPA}" \
              --max_segments "${MAX_SEGMENTS}"
        done
        
        python aggregate_evaluations.py \
          --input_files "${TEMP_OUTPUTS}" \
          --output ${EVALUATION_OUTPUT}
        
        # Step 5
        echo -e "\n[Step 5] Calculating paragraph-level metrics..."
        TEMP_OUTPUTS=""
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            TEMP_OUTPUT="${OUTPUT_DIR}/paragraph_${TARGET_DPA//' '/_}.json"
            if [ -z "$TEMP_OUTPUTS" ]; then
                TEMP_OUTPUTS="${TEMP_OUTPUT}"
            else
                TEMP_OUTPUTS="${TEMP_OUTPUTS},${TEMP_OUTPUT}"
            fi
            
            echo "Processing DPA: ${TARGET_DPA}"
            python paragraph_metrics.py \
              --results ${DEOLINGO_RESULTS} \
              --dpa ${DPA_CSV} \
              --evaluation ${EVALUATION_OUTPUT} \
              --output ${TEMP_OUTPUT} \
              --target_dpa "${TARGET_DPA}" \
              --max_segments "${MAX_SEGMENTS}"
        done
        
        python aggregate_paragraph_metrics.py \
          --input_files "${TEMP_OUTPUTS}" \
          --output ${PARAGRAPH_OUTPUT}
        
        echo -e "\n=============================================="
        echo "NEW PROMPTS TEST COMPLETED SUCCESSFULLY!"
        echo "=============================================="
        echo "Results summary:"
        echo "- DPA tested: ${TARGET_DPAS[*]}"
        echo "- Requirements tested: ${REQ_IDS}"
        echo "- Segments processed: ${MAX_SEGMENTS}"
        echo "- Deolingo results: ${DEOLINGO_RESULTS}"
        echo "- Evaluation results: ${EVALUATION_OUTPUT}"
        echo "- Paragraph metrics: ${PARAGRAPH_OUTPUT}"
        echo "=============================================="
        ;;
    Q|q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option. Please choose a valid step (1-5, A, or Q)."
        exit 1
        ;;
esac 