#!/bin/bash
# run_dpa_completeness_gpt4o.sh
# Master script to run the DPA completeness evaluation pipeline with GPT-4o
# Evaluates completeness of Online 1 DPA against requirements R7-R25

set -e  # Exit on any error

# Configuration
DPA_CSV="data/test_set.csv"
REQUIREMENTS_FILE="data/requirements/ground_truth_requirements.txt"
MODEL="gpt-4o-mini"  # GPT-4o model via API
OUTPUT_DIR="results/gpt4o_experiment/experiment_req"
DEOLINGO_RESULTS="${OUTPUT_DIR}/deolingo_results.txt"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.json"
PARAGRAPH_OUTPUT="${OUTPUT_DIR}/paragraph_metrics.json"
REQUIREMENTS_DEONTIC="results/requirements_deontic_ai_generated.json"
TARGET_DPAS=("Online 124")  # Array of DPAs to process
REQ_IDS="all"  # Focus on all requirements by default
MAX_SEGMENTS=0  # Process all segments by default

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
    --target_dpas)
      IFS=',' read -ra TARGET_DPAS <<< "$2"
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

echo "========== DPA Completeness Checker =========="
echo "Using GPT-4o Model via API"
echo "Using Deontic Logic and Answer Set Programming"
echo "Evaluating DPAs: ${TARGET_DPAS[*]}"
echo "Focus on requirement(s): ${REQ_IDS}"
echo "Using ${MAX_SEGMENTS} segments (0 means all)"
echo "=============================================="

# Show menu of available steps
echo "Available steps:"
echo "1. Translate all requirements to deontic logic"
echo "2. Generate LP files for specified requirements and segments for all DPAs"
echo "3. Run Deolingo solver for all DPAs"
echo "4. Evaluate DPA completeness (aggregated results)"
echo "5. Calculate paragraph-level metrics (aggregated results)"
echo "6. Debug requirement 28 segment 69"
echo "A. Run all steps sequentially"
echo "Q. Quit"

read -p "Enter step to run (1-6, A for all, Q to quit): " STEP

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
        echo -e "\n[Step 1] Translating all requirements to deontic logic..."
        python translate_requirements.py \
          --requirements ${REQUIREMENTS_FILE} \
          --model ${MODEL} \
          --output "${REQUIREMENTS_DEONTIC}"
        echo "Step 1 completed. Output saved to: ${REQUIREMENTS_DEONTIC}"
        ;;
    2)
        echo -e "\n[Step 2] Generating LP files for requirement(s) ${REQ_IDS} and ${MAX_SEGMENTS} segments..."
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            echo "Processing DPA: ${TARGET_DPA}"
            python generate_lp_files.py \
              --requirements "${REQUIREMENTS_DEONTIC}" \
              --dpa ${DPA_CSV} \
              --model ${MODEL} \
              --output "${OUTPUT_DIR}/lp_files" \
              --target_dpa "${TARGET_DPA}" \
              --max_segments ${MAX_SEGMENTS} \
              --req_ids "${REQ_IDS}"
        done
        echo "Step 2 completed. LP files generated in: ${OUTPUT_DIR}/lp_files"
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
    6)
        echo -e "\n[Step 6] Debugging requirement 28 segment 69..."
        
        # Create directory for requirement 28 if it doesn't exist
        TARGET_DPA="Online 124"
        REQ_DIR="${OUTPUT_DIR}/lp_files/dpa_${TARGET_DPA//' '/_}/req_28"
        mkdir -p "${REQ_DIR}"
        
        # Get the requirement text for R28 from requirements_deontic.json
        echo "Loading requirement text for R28 from ${REQUIREMENTS_DEONTIC}"
        REQ_TEXT=$(python -c "
import json
with open('${REQUIREMENTS_DEONTIC}', 'r') as f:
    data = json.load(f)
print(data.get('28', {}).get('text', 'Requirement text not found'))
")
        echo "Requirement 28 text: ${REQ_TEXT}"
        
        # Get the DPA segment 69 text
        echo "Loading DPA segment 69 text from ${DPA_CSV}"
        SEGMENT_TEXT=$(python -c "
import pandas as pd
df = pd.read_csv('${DPA_CSV}')
segment = df[(df['DPA'] == '${TARGET_DPA}') & (df['ID'] == 69)]
if not segment.empty:
    print(segment.iloc[0]['Sentence'])
else:
    print('Segment not found')
")
        echo "Segment 69 text: ${SEGMENT_TEXT}"
        
        # Create a simple LP file for testing
        echo -e "\nCreating test LP file for requirement 28, segment 69..."
        TEST_LP_FILE="${REQ_DIR}/segment_69.lp"
        
        # Write the LP file
        cat > "${TEST_LP_FILE}" << EOF
% Requirement Text:
% ${REQ_TEXT}
%
% DPA Segment:
% ${SEGMENT_TEXT}
%
% 1. Normative layer
&obligatory{remain_fully_liable_sub_processors} :- role(processor).

% 2. Facts extracted from DPA segment
remain_fully_liable_sub_processors.
role(processor).

% 3. Map Deolingo's internal status atoms to our labels
status(satisfied)     :- &fulfilled_obligation{remain_fully_liable_sub_processors}.
status(violated)      :- &violated_obligation{remain_fully_liable_sub_processors}.
status(not_mentioned) :- &undetermined_obligation{remain_fully_liable_sub_processors}.

#show status/1.
EOF
        
        echo "Test LP file created at: ${TEST_LP_FILE}"
        
        # Run deolingo on the test file
        echo -e "\nRunning deolingo on test LP file..."
        DEOLINGO_TEST_RESULTS="${OUTPUT_DIR}/deolingo_test_results.txt"
        echo "" > ${DEOLINGO_TEST_RESULTS}
        run_deolingo "${TEST_LP_FILE}" "28" "69" "${TARGET_DPA}"
        
        echo "Debug output written to: ${DEOLINGO_TEST_RESULTS}"
        
        # Run evaluation on the updated deolingo results
        echo -e "\nRunning evaluation with updated deolingo results..."
        python evaluate_completeness.py \
          --results ${DEOLINGO_RESULTS} \
          --dpa ${DPA_CSV} \
          --output "${OUTPUT_DIR}/evaluation_debug.json" \
          --target_dpa "${TARGET_DPA}"
        
        echo "Debugging complete. Check ${OUTPUT_DIR}/evaluation_debug.json for updated results."
        ;;
    A|a)
        echo -e "\n[Running all steps sequentially]"
        
        # Step 1: Translate requirements to deontic logic
        echo -e "\n[Step 1] Translating all requirements to deontic logic..."
        python translate_requirements.py \
          --requirements ${REQUIREMENTS_FILE} \
          --model ${MODEL} \
          --output "${REQUIREMENTS_DEONTIC}"
        echo "Step 1 completed. Output saved to: ${REQUIREMENTS_DEONTIC}"
        
        # Step 2: Generate LP files
        echo -e "\n[Step 2] Generating LP files for requirement(s) ${REQ_IDS} and ${MAX_SEGMENTS} segments..."
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            echo "Processing DPA: ${TARGET_DPA}"
            python generate_lp_files.py \
              --requirements "${REQUIREMENTS_DEONTIC}" \
              --dpa ${DPA_CSV} \
              --model ${MODEL} \
              --output "${OUTPUT_DIR}/lp_files" \
              --target_dpa "${TARGET_DPA}" \
              --max_segments ${MAX_SEGMENTS} \
              --req_ids "${REQ_IDS}"
        done
        echo "Step 2 completed. LP files generated in: ${OUTPUT_DIR}/lp_files"
        
        # Step 3: Run Deolingo solver
        echo -e "\n[Step 3] Running Deolingo solver for all DPAs..."
        # Create output file for results
        echo "" > ${DEOLINGO_RESULTS}
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            DPA_DIR="${OUTPUT_DIR}/lp_files/dpa_${TARGET_DPA//' '/_}"
            
            if [ ! -d "${DPA_DIR}" ]; then
                echo "Error: LP files directory not found at ${DPA_DIR} for DPA ${TARGET_DPA}. Skipping."
                continue
            fi
            
            echo "Processing DPA: ${TARGET_DPA}"
            
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
        
        # Rest of steps follow similarly...
        # Step 4: Evaluate completeness
        echo -e "\n[Step 4] Evaluating DPA completeness (aggregated results)..."
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
        
        # Step 5: Calculate paragraph metrics
        echo -e "\n[Step 5] Calculating paragraph-level metrics (aggregated results)..."
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
        echo -e "\nAll steps completed successfully!"
        ;;
    Q|q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option. Exiting..."
        exit 1
        ;;
esac

echo "Done!" 