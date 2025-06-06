#!/bin/bash
# run_dpa_completeness_ollama_new_prompts.sh
# Master script to run the DPA completeness evaluation pipeline with Ollama models using new requirement-specific prompts

set -e  # Exit on any error

# Configuration
DPA_CSV="data/test_set.csv"
REQUIREMENTS_FILE="data/requirements/requirements_deontic_ai_generated.json"
OLLAMA_MODEL="gemma3:27b"  # Default Ollama model
OUTPUT_DIR="results/ollama_experiment/fixed_prompts/gemma3-27b"
DEOLINGO_RESULTS="${OUTPUT_DIR}/deolingo_results.txt"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.json"
PARAGRAPH_OUTPUT="${OUTPUT_DIR}/paragraph_metrics.json"
REQUIREMENTS_DEONTIC="${OUTPUT_DIR}/requirements_deontic_generated.json"
TARGET_DPAS=("Online 124" "Online 126"  "Online 132")  # Default DPA for testing new prompts
REQ_IDS="all"  # Focus on requirements 1-6 for testing
MAX_SEGMENTS=0  # Limit to 50 segments for testing
REQUIREMENTS_REPRESENTATION="deontic_ai"  # Default representation
USE_PREDEFINED=true  # Default to using predefined requirements
REQUIREMENT_PROMPTS="requirement_prompts.json"  # New requirement-specific prompts

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            OLLAMA_MODEL="$2"
            shift 2
            ;;
        --requirements)
            REQUIREMENTS_REPRESENTATION="$2"
            shift 2
            ;;
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
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --requirement_prompts)
            REQUIREMENT_PROMPTS="$2"
            shift 2
            ;;
        --use_generated)
            USE_PREDEFINED=false
            shift
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

# Function to log messages with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if Ollama is running
check_ollama() {
    if ! curl -f -s http://localhost:11434/api/tags > /dev/null; then
        log "ERROR: Ollama server is not running!"
        log "Please start Ollama server first"
        exit 1
    fi
    log "Ollama server is running"
}

# Function to pull model if not available
ensure_model() {
    local model=$1
    log "Checking if model ${model} is available..."
    
    if ! ollama list | grep -q "${model}"; then
        log "Model ${model} not found locally. Pulling from registry..."
        if ! ollama pull "${model}"; then
            log "ERROR: Failed to pull model ${model}"
            exit 1
        fi
        log "Successfully pulled model ${model}"
    else
        log "Model ${model} is available"
    fi
}

# Function to set requirements file based on representation
set_requirements_file() {
    if [ "$USE_PREDEFINED" = true ]; then
        # Use predefined requirements files
        case $REQUIREMENTS_REPRESENTATION in
            "deontic")
                REQUIREMENTS_FILE="data/requirements/requirements_deontic.json"
                ;;
            "deontic_ai")
                REQUIREMENTS_FILE="data/requirements/requirements_deontic_ai_generated.json"
                ;;
            "deontic_experiments")
                REQUIREMENTS_FILE="data/requirements/requirements_deontic_experiments.json"
                ;;
            *)
                log "ERROR: Unknown requirements representation: $REQUIREMENTS_REPRESENTATION"
                log "Available options: deontic, deontic_ai, deontic_experiments"
                exit 1
                ;;
        esac
        
        if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
            log "ERROR: Requirements file not found: $REQUIREMENTS_FILE"
            exit 1
        fi
        
        log "Using predefined requirements file: $REQUIREMENTS_FILE"
    else
        # Use generated requirements file (will be created in Step 1)
        REQUIREMENTS_FILE="$REQUIREMENTS_DEONTIC"
        log "Will use generated requirements file: $REQUIREMENTS_FILE"
        log "Note: You must run Step 1 first to generate the requirements"
    fi
}

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

# Function to check if deolingo is installed
check_deolingo() {
    if ! command -v deolingo &> /dev/null; then
        log "ERROR: deolingo command not found!"
        log "Deolingo is required to run the solver (steps 3-5)."
        log ""
        log "To install deolingo on your server, run:"
        log "  pip3 install git+https://github.com/ovidiomanteiga/deolingo.git@main"
        log ""
        log "Or install from the local repository:"
        log "  cd $(pwd) && pip3 install -e ./deolingo/"
        log ""
        log "For detailed installation instructions, see:"
        log "  docs/DEOLINGO_SERVER_SETUP.md"
        log ""
        log "After installation, make sure deolingo is in your PATH:"
        log "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        exit 1
    fi
    log "Deolingo is available"
}

# Create output directory with model name
mkdir -p "$OUTPUT_DIR"

# Set requirements file
set_requirements_file

# Update output paths based on configuration
MODEL_SAFE_NAME=$(echo "${OLLAMA_MODEL}" | tr ':' '_' | tr '/' '_')
DEOLINGO_RESULTS="${OUTPUT_DIR}/deolingo_results_${MODEL_SAFE_NAME}_new_prompts.txt"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results_${MODEL_SAFE_NAME}_new_prompts.json"
PARAGRAPH_OUTPUT="${OUTPUT_DIR}/paragraph_metrics_${MODEL_SAFE_NAME}_new_prompts.json"

echo "========== DPA Completeness Checker with Ollama - Fixed Prompts =========="
echo "Using Ollama Model: ${OLLAMA_MODEL}"
echo "Testing FIXED REQUIREMENT-SPECIFIC PROMPTS (with head predicates in satisfying examples)"
echo "Requirements Representation: ${REQUIREMENTS_REPRESENTATION}"
if [ "$USE_PREDEFINED" = true ]; then
    echo "Requirements Source: Predefined (${REQUIREMENTS_FILE})"
else
    echo "Requirements Source: Generated (will be created in Step 1)"
    echo "Generated requirements will be saved to: ${REQUIREMENTS_DEONTIC}"
fi
echo "Evaluating DPAs: ${TARGET_DPAS[*]}"
echo "Focus on requirement(s): ${REQ_IDS}"
echo "Using ${MAX_SEGMENTS} segments for testing"
echo "Requirement prompts file: ${REQUIREMENT_PROMPTS}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "========================================================================"

# Check if requirement prompts file exists
if [ ! -f "${REQUIREMENT_PROMPTS}" ]; then
    echo "Error: Requirement prompts file ${REQUIREMENT_PROMPTS} not found!"
    echo "Please run: python generate_requirement_prompts.py first"
    exit 1
fi

# Show menu of available steps
echo "Available steps:"
if [ "$USE_PREDEFINED" = true ]; then
    echo "1. Translate all requirements to deontic logic (Not needed - using predefined requirements)"
else
    echo "1. Translate all requirements to deontic logic (REQUIRED - will generate requirements for steps 2-5)"
fi
echo "2. Generate LP files using NEW requirement-specific prompts"
echo "3. Run Deolingo solver for all DPAs"
echo "4. Evaluate DPA completeness (aggregated results)"
echo "5. Calculate paragraph-level metrics (aggregated results)"
echo "A. Run all steps sequentially"
echo "Q. Quit"

read -p "Enter step to run (1-5, A for all, Q to quit): " STEP

case ${STEP} in
    1)
        if [ "$USE_PREDEFINED" = true ]; then
            echo -e "\n[Step 1] Requirements translation not needed."
            echo "Using predefined requirements files from data/requirements/ directory."
            echo "Available representations: deontic, deontic_ai, deontic_experiments"
            echo "Currently using: ${REQUIREMENTS_REPRESENTATION} (${REQUIREMENTS_FILE})"
            echo ""
            echo "To use generated requirements instead, run with --use_generated flag"
        else
            echo -e "\n[Step 1] Translating all requirements to deontic logic using ${OLLAMA_MODEL}..."
            echo "This will generate requirements that will be used by steps 2-5."
            
            # Check Ollama server and model availability
            check_ollama
            ensure_model "$OLLAMA_MODEL"
            
            # Use the ground truth requirements as input for translation
            INPUT_REQUIREMENTS="data/requirements/ground_truth_requirements.txt"
            if [[ ! -f "$INPUT_REQUIREMENTS" ]]; then
                log "ERROR: Input requirements file not found: $INPUT_REQUIREMENTS"
                exit 1
            fi
            
            echo "Input: ${INPUT_REQUIREMENTS}"
            echo "Output: ${REQUIREMENTS_DEONTIC}"
            
            python3 translate_requirements.py \
              --requirements "$INPUT_REQUIREMENTS" \
              --model "$OLLAMA_MODEL" \
              --output "$REQUIREMENTS_DEONTIC"
            
            echo "Step 1 completed. Generated requirements saved to: ${REQUIREMENTS_DEONTIC}"
            echo "Note: Steps 2-5 will now use the generated requirements file"
        fi
        ;;
    2)
        echo -e "\n[Step 2] Generating LP files using NEW requirement-specific prompts..."
        echo "Testing requirements: ${REQ_IDS}"
        echo "Testing ${MAX_SEGMENTS} segments from DPA: ${TARGET_DPAS[*]}"
        
        # Check if requirements file exists (especially important for generated requirements)
        if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
            if [ "$USE_PREDEFINED" = false ]; then
                log "ERROR: Generated requirements file not found: $REQUIREMENTS_FILE"
                log "Please run Step 1 first to generate the requirements"
            else
                log "ERROR: Predefined requirements file not found: $REQUIREMENTS_FILE"
            fi
            exit 1
        fi
        
        # Check Ollama server and model availability (only needed for LP generation)
        check_ollama
        ensure_model "$OLLAMA_MODEL"
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            echo "Processing DPA: ${TARGET_DPA}"
            python3 generate_lp_files.py \
              --requirements "${REQUIREMENTS_FILE}" \
              --dpa ${DPA_CSV} \
              --model ${OLLAMA_MODEL} \
              --output "${OUTPUT_DIR}/lp_files" \
              --target_dpa "${TARGET_DPA}" \
              --max_segments ${MAX_SEGMENTS} \
              --req_ids "${REQ_IDS}" \
              --requirement_prompts "${REQUIREMENT_PROMPTS}" \
              --use_ollama \
              --verbose
        done
        echo "Step 2 completed. LP files with NEW prompts generated in: ${OUTPUT_DIR}/lp_files"
        ;;
    3)
        echo -e "\n[Step 3] Running Deolingo solver for all DPAs..."
        
        # Check if deolingo is installed
        check_deolingo
        
        # Create output file for results
        echo "" > ${DEOLINGO_RESULTS}
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            DPA_DIR="${OUTPUT_DIR}/lp_files/dpa_${TARGET_DPA//' '/_}"
            
            if [ ! -d "${DPA_DIR}" ]; then
                echo "Error: LP files directory not found at ${DPA_DIR} for DPA ${TARGET_DPA}. Run Step 2 first."
                continue
            fi
            
            echo "Processing DPA: ${TARGET_DPA}"
            
            # Handle both specific requirement IDs and "all" case
            REQ_DIRS=""
            if [ "${REQ_IDS}" = "all" ]; then
                # Find all requirement directories automatically
                for req_dir in "${DPA_DIR}"/req_*; do
                    if [ -d "${req_dir}" ]; then
                        REQ_DIRS="${REQ_DIRS} ${req_dir}"
                    fi
                done
            else
                # Process only the specified requirements
                for REQ_ID in $(echo ${REQ_IDS} | tr ',' ' '); do
                    if [ -d "${DPA_DIR}/req_${REQ_ID}" ]; then
                        REQ_DIRS="${REQ_DIRS} ${DPA_DIR}/req_${REQ_ID}"
                    fi
                done
            fi
            
            if [ -z "${REQ_DIRS}" ]; then
                echo "  Warning: No requirement directories found in ${DPA_DIR}"
                continue
            fi
            
            for req_dir in ${REQ_DIRS}; do
                req_id=$(basename ${req_dir} | sed 's/req_//')
                echo "  Processing requirement ${req_id}..."
                find "${req_dir}" -name "*.lp" | while read lp_file; do
                    segment_id=$(basename $lp_file | sed 's/segment_//' | sed 's/\.lp//')
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
            python3 evaluate_completeness.py \
              --results ${DEOLINGO_RESULTS} \
              --dpa ${DPA_CSV} \
              --output ${TEMP_OUTPUT} \
              --target_dpa "${TARGET_DPA}" \
              --max_segments "${MAX_SEGMENTS}"
        done
        
        # Aggregate results
        python3 aggregate_evaluations.py \
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
            python3 paragraph_metrics.py \
              --results ${DEOLINGO_RESULTS} \
              --dpa ${DPA_CSV} \
              --evaluation ${EVALUATION_OUTPUT} \
              --output ${TEMP_OUTPUT} \
              --target_dpa "${TARGET_DPA}" \
              --max_segments "${MAX_SEGMENTS}"
        done
        
        # Aggregate results
        python3 aggregate_paragraph_metrics.py \
          --input_files "${TEMP_OUTPUTS}" \
          --output ${PARAGRAPH_OUTPUT}
        
        echo "Step 5 completed. Paragraph metrics saved to: ${PARAGRAPH_OUTPUT}"
        ;;
    A|a)
        echo -e "\nRunning all steps sequentially for NEW prompts testing with Ollama..."
        
        # Step 1 (only if using generated requirements)
        if [ "$USE_PREDEFINED" = false ]; then
            echo -e "\n[Step 1] Translating all requirements to deontic logic using ${OLLAMA_MODEL}..."
            echo "This will generate requirements that will be used by steps 2-5."
            
            # Check Ollama server and model availability
            check_ollama
            ensure_model "$OLLAMA_MODEL"
            
            # Use the ground truth requirements as input for translation
            INPUT_REQUIREMENTS="data/requirements/ground_truth_requirements.txt"
            if [[ ! -f "$INPUT_REQUIREMENTS" ]]; then
                log "ERROR: Input requirements file not found: $INPUT_REQUIREMENTS"
                exit 1
            fi
            
            echo "Input: ${INPUT_REQUIREMENTS}"
            echo "Output: ${REQUIREMENTS_DEONTIC}"
            
            python3 translate_requirements.py \
              --requirements "$INPUT_REQUIREMENTS" \
              --model "$OLLAMA_MODEL" \
              --output "$REQUIREMENTS_DEONTIC"
            
            echo "Step 1 completed. Generated requirements saved to: ${REQUIREMENTS_DEONTIC}"
        else
            echo -e "\n[Step 1] Skipped - using predefined requirements: ${REQUIREMENTS_FILE}"
        fi
        
        # Step 2 - NEW PROMPTS
        echo -e "\n[Step 2] Generating LP files with NEW requirement-specific prompts..."
        
        # Check Ollama server and model availability (only needed for LP generation)
        check_ollama
        ensure_model "$OLLAMA_MODEL"
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            echo "Processing DPA: ${TARGET_DPA}"
            python3 generate_lp_files.py \
              --requirements "${REQUIREMENTS_FILE}" \
              --dpa ${DPA_CSV} \
              --model ${OLLAMA_MODEL} \
              --output "${OUTPUT_DIR}/lp_files" \
              --target_dpa "${TARGET_DPA}" \
              --max_segments ${MAX_SEGMENTS} \
              --req_ids "${REQ_IDS}" \
              --requirement_prompts "${REQUIREMENT_PROMPTS}" \
              --use_ollama \
              --verbose
        done
        
        # Step 3
        echo -e "\n[Step 3] Running Deolingo solver..."
        
        # Check if deolingo is installed
        check_deolingo
        
        echo "" > ${DEOLINGO_RESULTS}
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            DPA_DIR="${OUTPUT_DIR}/lp_files/dpa_${TARGET_DPA//' '/_}"
            
            if [ ! -d "${DPA_DIR}" ]; then
                echo "Error: LP files directory not found at ${DPA_DIR} for DPA ${TARGET_DPA}. Run Step 2 first."
                continue
            fi
            
            echo "Processing DPA: ${TARGET_DPA}"
            
            # Handle both specific requirement IDs and "all" case
            REQ_DIRS=""
            if [ "${REQ_IDS}" = "all" ]; then
                # Find all requirement directories automatically
                for req_dir in "${DPA_DIR}"/req_*; do
                    if [ -d "${req_dir}" ]; then
                        REQ_DIRS="${REQ_DIRS} ${req_dir}"
                    fi
                done
            else
                # Process only the specified requirements
                for REQ_ID in $(echo ${REQ_IDS} | tr ',' ' '); do
                    if [ -d "${DPA_DIR}/req_${REQ_ID}" ]; then
                        REQ_DIRS="${REQ_DIRS} ${DPA_DIR}/req_${REQ_ID}"
                    fi
                done
            fi
            
            if [ -z "${REQ_DIRS}" ]; then
                echo "  Warning: No requirement directories found in ${DPA_DIR}"
                continue
            fi
            
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
            python3 evaluate_completeness.py \
              --results ${DEOLINGO_RESULTS} \
              --dpa ${DPA_CSV} \
              --output ${TEMP_OUTPUT} \
              --target_dpa "${TARGET_DPA}" \
              --max_segments "${MAX_SEGMENTS}"
        done
        
        python3 aggregate_evaluations.py \
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
            python3 paragraph_metrics.py \
              --results ${DEOLINGO_RESULTS} \
              --dpa ${DPA_CSV} \
              --evaluation ${EVALUATION_OUTPUT} \
              --output ${TEMP_OUTPUT} \
              --target_dpa "${TARGET_DPA}" \
              --max_segments "${MAX_SEGMENTS}"
        done
        
        python3 aggregate_paragraph_metrics.py \
          --input_files "${TEMP_OUTPUTS}" \
          --output ${PARAGRAPH_OUTPUT}
        
        echo -e "\n========================================================================"
        echo "NEW PROMPTS TEST WITH OLLAMA COMPLETED SUCCESSFULLY!"
        echo "========================================================================"
        echo "Configuration used:"
        echo "  Model: ${OLLAMA_MODEL} (Ollama)"
        echo "  NEW REQUIREMENT-SPECIFIC PROMPTS approach"
        echo "  Requirements representation: ${REQUIREMENTS_REPRESENTATION}"
        if [ "$USE_PREDEFINED" = false ]; then
            echo "  Requirements source: Generated in Step 1"
        else
            echo "  Requirements source: Predefined"
        fi
        echo "  Target DPAs: ${TARGET_DPAS[*]}"
        echo "  Requirement IDs: ${REQ_IDS}"
        echo "  Max segments: ${MAX_SEGMENTS}"
        echo "  Requirement prompts: ${REQUIREMENT_PROMPTS}"
        echo ""
        echo "Results saved to:"
        echo "  Output directory: ${OUTPUT_DIR}"
        echo "  Deolingo results: ${DEOLINGO_RESULTS}"
        echo "  Evaluation results: ${EVALUATION_OUTPUT}"
        echo "  Paragraph metrics: ${PARAGRAPH_OUTPUT}"
        if [ "$USE_PREDEFINED" = false ]; then
            echo "  Generated requirements: ${REQUIREMENTS_DEONTIC}"
        fi
        echo "========================================================================"
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
