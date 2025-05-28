#!/bin/bash
# run_dpa_completeness_ollama.sh
# Master script to run the DPA completeness evaluation pipeline with Ollama models

set -e  # Exit on any error

# Configuration
DPA_CSV="data/test_set.csv"
REQUIREMENTS_FILE="data/requirements/requirements_deontic_ai_generated.json"
OLLAMA_MODEL="llama3.3:70b"  # Default Ollama model
OUTPUT_DIR="results/ollama_experiments"
DEOLINGO_RESULTS="${OUTPUT_DIR}/deolingo_results.txt"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.json"
PARAGRAPH_OUTPUT="${OUTPUT_DIR}/paragraph_metrics.json"
REQUIREMENTS_DEONTIC="results/requirements_deontic_ai_generated.json"
TARGET_DPAS=("Online 124")  # Array of DPAs to process
REQ_IDS="all"  # Focus on all requirements by default
MAX_SEGMENTS=0  # Process all segments by default
REQUIREMENTS_REPRESENTATION="deontic_ai"  # Default representation

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
        --req-ids)
            REQ_IDS="$2"
            shift 2
            ;;
        --max-segments)
            MAX_SEGMENTS="$2"
            shift 2
            ;;
        --target-dpa)
            TARGET_DPAS=("$2")
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL                Ollama model to use (default: llama3.3:70b)"
            echo "  --requirements REPR          Requirements representation (deontic, deontic_ai, deontic_experiments)"
            echo "  --req-ids IDS               Comma-separated requirement IDs or 'all'"
            echo "  --max-segments N            Maximum segments to process (0 for all)"
            echo "  --target-dpa DPA            Target DPA to process"
            echo "  --output-dir DIR            Output directory"
            echo "  --help                      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
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
    log "Checking Ollama server status..."
    if ! curl -f -s http://localhost:11434/api/tags > /dev/null; then
        log "ERROR: Ollama server is not running!"
        log "Please start Ollama server first with: ollama serve"
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

# Create output directory
log "Creating output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Check Ollama server and model availability
check_ollama
ensure_model "$OLLAMA_MODEL"

# Set requirements file
set_requirements_file

# Update output paths based on configuration
DEOLINGO_RESULTS="${OUTPUT_DIR}/deolingo_results_${OLLAMA_MODEL//[^a-zA-Z0-9]/_}_${REQUIREMENTS_REPRESENTATION}.txt"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results_${OLLAMA_MODEL//[^a-zA-Z0-9]/_}_${REQUIREMENTS_REPRESENTATION}.json"
PARAGRAPH_OUTPUT="${OUTPUT_DIR}/paragraph_metrics_${OLLAMA_MODEL//[^a-zA-Z0-9]/_}_${REQUIREMENTS_REPRESENTATION}.json"

log "Starting DPA completeness evaluation pipeline"
log "Configuration:"
log "  Model: ${OLLAMA_MODEL}"
log "  Requirements: ${REQUIREMENTS_REPRESENTATION} (${REQUIREMENTS_FILE})"
log "  Target DPAs: ${TARGET_DPAS[*]}"
log "  Requirement IDs: ${REQ_IDS}"
log "  Max segments: ${MAX_SEGMENTS}"
log "  Output directory: ${OUTPUT_DIR}"

# Step 1: Generate LP files using Ollama
for DPA in "${TARGET_DPAS[@]}"; do
    log "Step 1: Generating LP files for DPA: ${DPA}"
    
    LP_OUTPUT_DIR="${OUTPUT_DIR}/lp_files_${DPA// /_}"
    
    python3 generate_lp_files.py \
        --requirements "${REQUIREMENTS_FILE}" \
        --dpa "${DPA_CSV}" \
        --model "${OLLAMA_MODEL}" \
        --output "${LP_OUTPUT_DIR}" \
        --target_dpa "${DPA}" \
        --req_ids "${REQ_IDS}" \
        --max_segments "${MAX_SEGMENTS}" \
        --use_ollama \
        --verbose
    
    if [[ $? -ne 0 ]]; then
        log "ERROR: LP file generation failed for DPA: ${DPA}"
        exit 1
    fi
    
    log "LP files generated successfully for DPA: ${DPA}"
    
    # Step 2: Run Deolingo on generated LP files
    log "Step 2: Running Deolingo on LP files for DPA: ${DPA}"
    
    python3 run_deolingo.py \
        --input_dir "${LP_OUTPUT_DIR}" \
        --output "${DEOLINGO_RESULTS}" \
        --verbose
    
    if [[ $? -ne 0 ]]; then
        log "ERROR: Deolingo execution failed for DPA: ${DPA}"
        exit 1
    fi
    
    log "Deolingo execution completed for DPA: ${DPA}"
done

# Step 3: Evaluate results
log "Step 3: Evaluating results"

python3 evaluate_completeness.py \
    --deolingo_results "${DEOLINGO_RESULTS}" \
    --dpa_csv "${DPA_CSV}" \
    --requirements "${REQUIREMENTS_FILE}" \
    --output "${EVALUATION_OUTPUT}" \
    --target_dpas "${TARGET_DPAS[*]}" \
    --verbose

if [[ $? -ne 0 ]]; then
    log "ERROR: Evaluation failed"
    exit 1
fi

log "Evaluation completed successfully"

# Step 4: Generate paragraph-level metrics
log "Step 4: Generating paragraph-level metrics"

python3 calculate_paragraph_metrics.py \
    --evaluation_results "${EVALUATION_OUTPUT}" \
    --output "${PARAGRAPH_OUTPUT}" \
    --verbose

if [[ $? -ne 0 ]]; then
    log "ERROR: Paragraph metrics calculation failed"
    exit 1
fi

log "Paragraph metrics calculated successfully"

# Step 5: Display summary
log "============================================"
log "EXPERIMENT COMPLETED SUCCESSFULLY"
log "============================================"
log "Configuration used:"
log "  Model: ${OLLAMA_MODEL}"
log "  Requirements representation: ${REQUIREMENTS_REPRESENTATION}"
log "  Target DPAs: ${TARGET_DPAS[*]}"
log "  Requirement IDs: ${REQ_IDS}"
log "  Max segments: ${MAX_SEGMENTS}"
log ""
log "Results saved to:"
log "  Deolingo results: ${DEOLINGO_RESULTS}"
log "  Evaluation results: ${EVALUATION_OUTPUT}"
log "  Paragraph metrics: ${PARAGRAPH_OUTPUT}"
log ""
log "To view results:"
log "  cat ${EVALUATION_OUTPUT}"
log "  cat ${PARAGRAPH_OUTPUT}"
log "============================================" 