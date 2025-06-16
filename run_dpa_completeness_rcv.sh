#!/bin/bash
# run_dpa_completeness_rcv.sh
# Master script to run the DPA completeness evaluation pipeline with RCV approach

set -e  # Exit on any error

# Configuration
DPA_CSV="data/test_set.csv"
REQUIREMENTS_FILE="data/requirements/requirements_deontic_ai_generated.json"
OLLAMA_MODEL="qwen2.5:32b"  # Default Ollama model for verification step
EMBEDDING_MODEL="all-MiniLM-L6-v2"  # Default embedding model for classification step  
SIMILARITY_THRESHOLD=0.5  # Default similarity threshold (optimized from training data)
OUTPUT_DIR="results/rcv_approach/embeddings/qwen2-5"
DEOLINGO_RESULTS="${OUTPUT_DIR}/deolingo_results.txt"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.json"
PARAGRAPH_OUTPUT="${OUTPUT_DIR}/paragraph_metrics.json"
TARGET_DPAS=("Online 124" "Online 132" "Online 54")  # Array of DPAs to process
MAX_SEGMENTS=0  # Process all segments by default
REQUIREMENTS_REPRESENTATION="deontic_ai"  # Default representation
USE_PREDEFINED=true  # Default to using predefined requirements

# Default step (empty means interactive)
STEP=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --step)
            STEP="$2"
            shift 2
            ;;
        --model)
            OLLAMA_MODEL="$2"
            shift 2
            ;;
        --embedding_model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --similarity_threshold)
            SIMILARITY_THRESHOLD="$2"
            shift 2
            ;;
        --requirements)
            REQUIREMENTS_REPRESENTATION="$2"
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
        --debug)
            DEBUG=true
            shift
            ;;
        --help|-h)
            echo "DPA Completeness Checker - RCV Approach (Embedding-Based)"
            echo ""
            echo "Usage: $0 [STEP] [OPTIONS]"
            echo ""
            echo "Steps:"
            echo "  --step 1                      Generate RCV LP files"
            echo "  --step 2                      Run Deolingo solver" 
            echo "  --step 3                      Evaluate completeness"
            echo "  --step A                      Run all steps sequentially"
            echo "  (no --step)                   Interactive mode"
            echo ""
            echo "Options:"
            echo "  --model MODEL                 Ollama model for verification step (default: qwen3:32b)"
            echo "  --embedding_model MODEL       Sentence transformer model for classification (default: all-MiniLM-L6-v2)" 
            echo "  --similarity_threshold FLOAT  Cosine similarity threshold for classification (default: 0.95)"
            echo "  --requirements REPR           Requirements representation (deontic, deontic_ai, deontic_experiments)"
            echo "  --max_segments N              Maximum segments to process (0 means all)"
            echo "  --target_dpas DPA1,DPA2       Comma-separated list of DPAs to process"
            echo "  --output_dir DIR              Output directory for results"
            echo "  --debug                       Enable debug mode"
            echo "  --help, -h                    Show this help message"
            echo ""
            echo "Embedding Model Options:"
            echo "  - all-MiniLM-L6-v2            Fast, general-purpose (default)"
            echo "  - all-mpnet-base-v2           Better quality, slower"
            echo "  - sentence-transformers/all-roberta-large-v1  High quality, slower"
            echo ""
            echo "Similarity Threshold Guidelines:"
            echo "  - 0.95: High precision, low recall (~0.3% classification rate)"
            echo "  - 0.70-0.80: Balanced precision/recall (~4-17% classification rate)"
            echo "  - 0.30-0.40: High recall, low precision (~78-91% classification rate)"
            exit 0
            ;;
        1|2|3|A|a)
            # Allow step as first positional argument (backward compatibility)
            STEP="$1"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help for usage information"
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
    
    log "Using requirements file: $REQUIREMENTS_FILE"
}

# Function to map requirement numbers to R-labels for evaluation compatibility
map_req_to_r_label() {
    local req_num=$1
    case ${req_num} in
        1) echo "10" ;;
        2) echo "11" ;;
        3) echo "12" ;;
        4) echo "13" ;;
        5) echo "15" ;;
        6) echo "16" ;;
        7) echo "17" ;;
        8) echo "18" ;;
        9) echo "19" ;;
        10) echo "20" ;;
        11) echo "21" ;;
        12) echo "22" ;;
        13) echo "23" ;;
        14) echo "24" ;;
        15) echo "25" ;;
        16) echo "26" ;;
        17) echo "27" ;;
        18) echo "28" ;;
        19) echo "29" ;;
        *) echo "${req_num}" ;;
    esac
}

# Function to run deolingo with error handling (updated for requirement-specific processing)
run_deolingo() {
    local lp_file=$1
    local req_id=$2
    local segment_id=$3
    local dpa_name=$4
    
    # Map requirement number to R-label for evaluation compatibility
    local r_label=$(map_req_to_r_label ${req_id})
    
    # Run deolingo and capture output
    deolingo_output=$(deolingo ${lp_file} 2>&1) || true
    
    # Check if there was an error
    if [[ $deolingo_output == *"ERROR"* || $deolingo_output == *"error"* ]]; then
        echo "Processing DPA ${dpa_name}, Requirement ${r_label}, Segment ${segment_id}..." >> ${DEOLINGO_RESULTS}
        echo "Error processing file: ${lp_file}" >> ${DEOLINGO_RESULTS}
        echo "Answer: not_mentioned" >> ${DEOLINGO_RESULTS}
        echo "--------------------------------------------------" >> ${DEOLINGO_RESULTS}
        echo "Warning: Error in file ${lp_file}. Skipping and continuing..." >&2
    else
        echo "Processing DPA ${dpa_name}, Requirement ${r_label}, Segment ${segment_id}..." >> ${DEOLINGO_RESULTS}
        echo "${deolingo_output}" >> ${DEOLINGO_RESULTS}
        echo "--------------------------------------------------" >> ${DEOLINGO_RESULTS}
    fi
}

# Function to check if deolingo is installed
check_deolingo() {
    if ! command -v deolingo &> /dev/null; then
        log "ERROR: deolingo command not found!"
        log "Deolingo is required to run the solver (steps 2-4)."
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

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set requirements file
set_requirements_file

# Update output paths based on configuration
MODEL_SAFE_NAME=$(echo "${OLLAMA_MODEL}" | tr ':' '_' | tr '/' '_')
DEOLINGO_RESULTS="${OUTPUT_DIR}/deolingo_results_${MODEL_SAFE_NAME}_rcv.txt"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results_${MODEL_SAFE_NAME}_rcv.json"

echo "========== DPA Completeness Checker - RCV Approach (Embedding-Based) =========="
echo "Verification Model (LLM): ${OLLAMA_MODEL}"
echo "Classification Model (Embedding): ${EMBEDDING_MODEL}"
echo "Similarity Threshold: ${SIMILARITY_THRESHOLD}"
echo "Requirements Representation: ${REQUIREMENTS_REPRESENTATION}"
echo "Requirements Source: ${REQUIREMENTS_FILE}"
echo "Evaluating DPAs: ${TARGET_DPAS[*]}"
echo "Using ${MAX_SEGMENTS} segments (0 means all)"
echo "Output Directory: ${OUTPUT_DIR}"
echo "========================================================"

# Show menu of available steps (only if STEP not provided via command line)
if [[ -z "$STEP" ]]; then
    echo "Available steps:"
    echo "1. Generate RCV LP files (embedding-based classification + LLM verification)"
    echo "2. Run Deolingo solver for all DPAs"
    echo "3. Evaluate DPA completeness (Requirement & Segment-level metrics)"
    echo "A. Run all steps sequentially"
    echo "Q. Quit"

    read -p "Enter step to run (1-3, A for all, Q to quit): " STEP
fi

case ${STEP} in
    1)
        echo -e "\n[Step 1] Generating RCV LP files for ${MAX_SEGMENTS} segments..."
        
        # Check if requirements file exists
        if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
            log "ERROR: Requirements file not found: $REQUIREMENTS_FILE"
            exit 1
        fi
        
        # Check Ollama server and model availability (for verification step)
        check_ollama
        ensure_model "$OLLAMA_MODEL"
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            echo "Processing DPA: ${TARGET_DPA}"
            python3 generate_rcv_lp_files.py \
              --requirements "${REQUIREMENTS_FILE}" \
              --dpa_segments ${DPA_CSV} \
              --model ${OLLAMA_MODEL} \
              --embedding_model ${EMBEDDING_MODEL} \
              --similarity_threshold ${SIMILARITY_THRESHOLD} \
              --output "${OUTPUT_DIR}/lp_files_${TARGET_DPA//' '/_}" \
              --target_dpa "${TARGET_DPA}" \
              --max_segments ${MAX_SEGMENTS} \
              --verbose
        done
        echo "Step 1 completed. RCV LP files generated in: ${OUTPUT_DIR}/lp_files_[DPA_NAME]"
        ;;
    2)
        echo -e "\n[Step 2] Running Deolingo solver for all DPAs..."
        
        # Check if deolingo is installed
        check_deolingo
        
        # Create output file for results
        echo "" > ${DEOLINGO_RESULTS}
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            DPA_DIR="${OUTPUT_DIR}/lp_files_${TARGET_DPA//' '/_}"
            
            if [ ! -d "${DPA_DIR}" ]; then
                echo "Error: LP files directory not found at ${DPA_DIR} for DPA ${TARGET_DPA}. Run Step 1 first."
                continue
            fi
            
            echo "Processing DPA: ${TARGET_DPA}"
            
            # Process all .lp files for this DPA (now using req_* subdirectories like original approach)
            find "${DPA_DIR}" -path "*/req_*/segment_*.lp" | while read lp_file; do
                # Extract requirement ID and segment ID from the file path
                req_id=$(echo $lp_file | sed 's/.*req_\([0-9]*\).*/\1/')
                segment_id=$(basename $lp_file | sed 's/segment_//' | sed 's/\.lp//')
                
                # Run deolingo with error handling
                run_deolingo "${lp_file}" "${req_id}" "${segment_id}" "${TARGET_DPA}"
            done
        done
        echo "Step 2 completed. Results saved in: ${DEOLINGO_RESULTS}"
        ;;
    3)
        echo -e "\n[Step 3] Evaluating DPA completeness (aggregated results)..."
        if [ ! -f "${DEOLINGO_RESULTS}" ]; then
            echo "Error: Deolingo results file not found. Run Step 2 first."
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
        
        echo "Step 3 completed. Aggregated evaluation results saved to: ${EVALUATION_OUTPUT}"
        ;;

    A|a)
        echo -e "\nRunning all steps sequentially..."
        
        # Step 1
        echo -e "\n[Step 1] Generating RCV LP files for ${MAX_SEGMENTS} segments..."
        
        # Check Ollama server and model availability (for verification step)
        check_ollama
        ensure_model "$OLLAMA_MODEL"
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            echo "Processing DPA: ${TARGET_DPA}"
            python3 generate_rcv_lp_files.py \
              --requirements "${REQUIREMENTS_FILE}" \
              --dpa_segments ${DPA_CSV} \
              --model ${OLLAMA_MODEL} \
              --embedding_model ${EMBEDDING_MODEL} \
              --similarity_threshold ${SIMILARITY_THRESHOLD} \
              --output "${OUTPUT_DIR}/lp_files_${TARGET_DPA//' '/_}" \
              --target_dpa "${TARGET_DPA}" \
              --max_segments ${MAX_SEGMENTS} \
              --verbose
        done
        echo "Step 1 completed. RCV LP files generated."
        
        # Step 2
        echo -e "\n[Step 2] Running Deolingo solver for all DPAs..."
        check_deolingo
        echo "" > ${DEOLINGO_RESULTS}
        
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            DPA_DIR="${OUTPUT_DIR}/lp_files_${TARGET_DPA//' '/_}"
            echo "Processing DPA: ${TARGET_DPA}"
            
            find "${DPA_DIR}" -path "*/req_*/segment_*.lp" | while read lp_file; do
                req_id=$(echo $lp_file | sed 's/.*req_\([0-9]*\).*/\1/')
                segment_id=$(basename $lp_file | sed 's/segment_//' | sed 's/\.lp//')
                run_deolingo "${lp_file}" "${req_id}" "${segment_id}" "${TARGET_DPA}"
            done
        done
        echo "Step 2 completed. Deolingo results saved."
        
        # Step 3
        echo -e "\n[Step 3] Evaluating DPA completeness..."
        TEMP_OUTPUTS=""
        for TARGET_DPA in "${TARGET_DPAS[@]}"; do
            TEMP_OUTPUT="${OUTPUT_DIR}/evaluation_${TARGET_DPA//' '/_}.json"
            if [ -z "$TEMP_OUTPUTS" ]; then
                TEMP_OUTPUTS="${TEMP_OUTPUT}"
            else
                TEMP_OUTPUTS="${TEMP_OUTPUTS},${TEMP_OUTPUT}"
            fi
            
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
        
        echo -e "\n========================================================"
        echo "All steps completed successfully!"
        echo "Final results:"
        echo "  - Deolingo results: ${DEOLINGO_RESULTS}"
        echo "  - Evaluation results: ${EVALUATION_OUTPUT}"
        echo "========================================================"
        ;;
    Q|q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option: ${STEP}"
        echo "Please choose 1-3, A for all, or Q to quit"
        exit 1
        ;;
esac

echo "Script completed successfully!" 
