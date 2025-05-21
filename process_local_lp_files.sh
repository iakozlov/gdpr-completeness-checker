#!/bin/bash
# process_local_lp_files.sh
# Script to process locally downloaded .lp files, run deolingo, and evaluate results

set -e  # Exit on any error

# Configuration
INPUT_DIR="results/Qwen"  # Where you'll manually upload the downloaded files
OUTPUT_DIR="results/Qwen"
DEOLINGO_RESULTS="${OUTPUT_DIR}/deolingo_results.txt"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.json"
PARAGRAPH_OUTPUT="${OUTPUT_DIR}/paragraph_metrics.json"
DPA_CSV="data/test_set.csv"
MAX_SEGMENTS=0  # Process all segments by default
DPA_FOLDER="Online_124"  # Default DPA folder

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dpa_folder)
      DPA_FOLDER="$2"
      shift 2
      ;;
    --max_segments)
      MAX_SEGMENTS="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Check if input directory exists
if [ ! -d "${INPUT_DIR}/${DPA_FOLDER}" ]; then
  echo "Error: Input directory '${INPUT_DIR}/${DPA_FOLDER}' not found."
  echo "Please download the LP files from Google Drive and place them in ${INPUT_DIR}/${DPA_FOLDER}"
  echo "You can specify a different folder using --dpa_folder parameter"
  exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "========== DPA Completeness Checker =========="
echo "Using Deontic Logic and Answer Set Programming"
echo "Processing DPA from local folder: ${DPA_FOLDER}"
echo "Using ${MAX_SEGMENTS} segments (0 means all)"
echo "============================================="

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

# Show menu of available steps
echo "Available steps:"
echo "1. Run Deolingo solver for local LP files"
echo "2. Evaluate DPA completeness"
echo "3. Calculate paragraph-level metrics"
echo "A. Run all steps sequentially"
echo "Q. Quit"

read -p "Enter step to run (1-3, A for all, Q to quit): " STEP

# Determine DPA name from folder name
DPA_NAME="${DPA_FOLDER//_/ }"  # Replace underscores with spaces

case ${STEP} in
    1)
        echo -e "\n[Step 1] Running Deolingo solver for local LP files..."
        
        # Create output file for results
        echo "" > ${DEOLINGO_RESULTS}
        
        # Process all req_* directories
        REQ_DIRS=$(find "${INPUT_DIR}/${DPA_FOLDER}" -type d -name "req_*")
        for req_dir in ${REQ_DIRS}; do
            req_id=$(basename ${req_dir} | sed 's/req_//')
            echo "  Processing requirement ${req_id}..."
            
            # Process all .lp files for this requirement
            find "${req_dir}" -name "*.lp" | while read lp_file; do
                # Extract segment ID from the file path
                segment_id=$(basename $lp_file | sed 's/segment_//' | sed 's/\.lp//')
                
                # Apply max_segments limit if set
                if [ ${MAX_SEGMENTS} -gt 0 ] && [ ${segment_id} -gt ${MAX_SEGMENTS} ]; then
                    continue
                fi
                
                # Run deolingo with error handling
                run_deolingo "${lp_file}" "${req_id}" "${segment_id}" "${DPA_NAME}"
            done
        done
        
        echo "Step 1 completed. Results saved in: ${DEOLINGO_RESULTS}"
        ;;
    2)
        echo -e "\n[Step 2] Evaluating DPA completeness..."
        if [ ! -f "${DEOLINGO_RESULTS}" ]; then
            echo "Error: Deolingo results file not found. Run Step 1 first."
            exit 1
        fi
        
        python evaluate_completeness.py \
          --results ${DEOLINGO_RESULTS} \
          --dpa ${DPA_CSV} \
          --output ${EVALUATION_OUTPUT} \
          --target_dpa "${DPA_NAME}" \
          --max_segments "${MAX_SEGMENTS}"
        echo "Step 2 completed. Evaluation results saved to: ${EVALUATION_OUTPUT}"
        ;;
    3)
        echo -e "\n[Step 3] Calculating paragraph-level metrics..."
        if [ ! -f "${DEOLINGO_RESULTS}" ]; then
            echo "Error: Deolingo results file not found. Run Step 1 first."
            exit 1
        fi
        if [ ! -f "${EVALUATION_OUTPUT}" ]; then
            echo "Error: Evaluation results file not found. Run Step 2 first."
            exit 1
        fi
        
        python paragraph_metrics.py \
          --results ${DEOLINGO_RESULTS} \
          --dpa ${DPA_CSV} \
          --evaluation ${EVALUATION_OUTPUT} \
          --output ${PARAGRAPH_OUTPUT} \
          --target_dpa "${DPA_NAME}"
        echo "Step 3 completed. Paragraph metrics saved to: ${PARAGRAPH_OUTPUT}"
        ;;
    [aA])
        echo -e "\nRunning all steps sequentially..."
        
        echo -e "\n[Step 1] Running Deolingo solver for local LP files..."
        # Create output file for results
        echo "" > ${DEOLINGO_RESULTS}
        
        # Process all req_* directories
        REQ_DIRS=$(find "${INPUT_DIR}/${DPA_FOLDER}" -type d -name "req_*")
        for req_dir in ${REQ_DIRS}; do
            req_id=$(basename ${req_dir} | sed 's/req_//')
            echo "  Processing requirement ${req_id}..."
            
            # Process all .lp files for this requirement
            find "${req_dir}" -name "*.lp" | while read lp_file; do
                # Extract segment ID from the file path
                segment_id=$(basename $lp_file | sed 's/segment_//' | sed 's/\.lp//')
                
                # Apply max_segments limit if set
                if [ ${MAX_SEGMENTS} -gt 0 ] && [ ${segment_id} -gt ${MAX_SEGMENTS} ]; then
                    continue
                fi
                
                # Run deolingo with error handling
                run_deolingo "${lp_file}" "${req_id}" "${segment_id}" "${DPA_NAME}"
            done
        done
        
        echo -e "\n[Step 2] Evaluating DPA completeness..."
        python evaluate_completeness.py \
          --results ${DEOLINGO_RESULTS} \
          --dpa ${DPA_CSV} \
          --output ${EVALUATION_OUTPUT} \
          --target_dpa "${DPA_NAME}" \
          --max_segments "${MAX_SEGMENTS}"
        
        echo -e "\n[Step 3] Calculating paragraph-level metrics..."
        python paragraph_metrics.py \
          --results ${DEOLINGO_RESULTS} \
          --dpa ${DPA_CSV} \
          --evaluation ${EVALUATION_OUTPUT} \
          --output ${PARAGRAPH_OUTPUT} \
          --target_dpa "${DPA_NAME}"
        
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