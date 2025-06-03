#!/bin/bash
# run_baseline_evaluation.sh
# Baseline evaluation script for DPA completeness using direct LLM classification

set -e  # Exit on any error

# Configuration
DPA_CSV="data/train_set.csv"
OLLAMA_MODEL="llama3.3:70b"  # Default Ollama model
OUTPUT_DIR="results/baseline_evaluation"
BASELINE_RESULTS="${OUTPUT_DIR}/baseline_deolingo_format.txt"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results.json"
PARAGRAPH_OUTPUT="${OUTPUT_DIR}/paragraph_metrics.json"
TARGET_DPAS=("Online 124" "Online 132")  # Array of DPAs to process
REQ_IDS="all"  # Focus on all requirements by default
MAX_SEGMENTS=0  # Process all segments by default
SAMPLE_RATIO=0.1  # Sample 10% of segments for baseline
SEED=42  # Random seed for reproducible sampling
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            OLLAMA_MODEL="$2"
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
        --sample_ratio)
            SAMPLE_RATIO="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Baseline DPA Completeness Evaluation using Direct LLM Classification"
            echo ""
            echo "Options:"
            echo "  --model MODEL      Ollama model to use (default: llama3.3:70b)"
            echo "  --req_ids IDS      Comma-separated requirement IDs or 'all' (default: all)"
            echo "  --max_segments N  Maximum segments per DPA (0=all, default: 0)"
            echo "  --target_dpas DPAS Comma-separated target DPAs (default: 'Online 124,Online 132')"
            echo "  --output_dir DIR  Output directory (default: results/baseline_evaluation)"
            echo "  --sample_ratio RATIO Sampling ratio for baseline (default: 0.1)"
            echo "  --seed SEED       Random seed for reproducibility (default: 42)"
            echo "  --verbose         Enable verbose output"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model llama3.3:70b --target_dpas 'Online 124'"
            echo "  $0 --req_ids '1,2,3' --sample_ratio 0.2 --verbose"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
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

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Update output paths based on configuration
MODEL_SAFE_NAME=$(echo "${OLLAMA_MODEL}" | tr ':' '_' | tr '/' '_')
BASELINE_RESULTS="${OUTPUT_DIR}/baseline_deolingo_format_${MODEL_SAFE_NAME}.txt"
EVALUATION_OUTPUT="${OUTPUT_DIR}/evaluation_results_${MODEL_SAFE_NAME}.json"
PARAGRAPH_OUTPUT="${OUTPUT_DIR}/paragraph_metrics_${MODEL_SAFE_NAME}.json"

echo "=========================================="
echo "BASELINE DPA COMPLETENESS EVALUATION"
echo "=========================================="
echo "Model: $OLLAMA_MODEL"
echo "DPA File: $DPA_CSV"
echo "Target DPAs: ${TARGET_DPAS[*]}"
echo "Requirements: $REQ_IDS"
echo "Max segments: $MAX_SEGMENTS"
echo "Sample ratio: $SAMPLE_RATIO"
echo "Random seed: $SEED"
echo "Results directory: $OUTPUT_DIR"
echo "=========================================="

# Show menu of available steps
echo "Available steps:"
echo "1. Run baseline LLM classification on sampled DPA segments"
echo "2. Evaluate completeness using baseline results (aggregated results)"
echo "3. Calculate paragraph-level metrics using baseline results (aggregated results)"
echo "A. Run all steps sequentially"
echo "Q. Quit"

read -p "Enter step to run (1-3, A for all, Q to quit): " STEP

case ${STEP} in
    1)
        echo -e "\n[Step 1] Running baseline LLM classification using ${OLLAMA_MODEL}..."
        
        # Check Ollama server and model availability
        check_ollama
        ensure_model "$OLLAMA_MODEL"
        
        echo "Target DPAs: ${TARGET_DPAS[*]}"
        echo "Sample ratio: ${SAMPLE_RATIO}"
        echo "Output: ${BASELINE_RESULTS}"
        
        python3 baseline_evaluation.py \
          --model "${OLLAMA_MODEL}" \
          --dpa "${DPA_CSV}" \
          --output "${OUTPUT_DIR}" \
          --target_dpas "$(IFS=','; echo "${TARGET_DPAS[*]}")" \
          --req_ids "${REQ_IDS}" \
          --max_segments "${MAX_SEGMENTS}" \
          --sample_ratio "${SAMPLE_RATIO}" \
          --seed "${SEED}" \
          $([ "$VERBOSE" = true ] && echo "--verbose")
        
        echo "Step 1 completed. Baseline results saved in: ${OUTPUT_DIR}"
        ;;
    2)
        echo -e "\n[Step 2] Evaluating DPA completeness using baseline results (aggregated results)..."
        if [ ! -f "${BASELINE_RESULTS}" ]; then
            echo "Error: Baseline results file not found. Run Step 1 first."
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
              --results "${BASELINE_RESULTS}" \
              --dpa "${DPA_CSV}" \
              --output "${TEMP_OUTPUT}" \
              --target_dpa "${TARGET_DPA}" \
              --max_segments "${MAX_SEGMENTS}" \
              $([ "$VERBOSE" = true ] && echo "--debug")
        done
        
        # Aggregate results
        python3 aggregate_evaluations.py \
          --input_files "${TEMP_OUTPUTS}" \
          --output "${EVALUATION_OUTPUT}"
        
        echo "Step 2 completed. Aggregated evaluation results saved to: ${EVALUATION_OUTPUT}"
        ;;
    3)
        echo -e "\n[Step 3] Calculating paragraph-level metrics using baseline results (aggregated results)..."
        if [ ! -f "${BASELINE_RESULTS}" ]; then
            echo "Error: Baseline results file not found. Run Step 1 first."
            exit 1
        fi
        if [ ! -f "${EVALUATION_OUTPUT}" ]; then
            echo "Error: Evaluation results file not found. Run Step 2 first."
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
              --results "${BASELINE_RESULTS}" \
              --dpa "${DPA_CSV}" \
              --evaluation "${EVALUATION_OUTPUT}" \
              --output "${TEMP_OUTPUT}" \
              --target_dpa "${TARGET_DPA}" \
              --max_segments "${MAX_SEGMENTS}" \
              $([ "$VERBOSE" = true ] && echo "--debug")
        done
        
        # Aggregate results
        python3 aggregate_paragraph_metrics.py \
          --input_files "${TEMP_OUTPUTS}" \
          --output "${PARAGRAPH_OUTPUT}"
        
        echo "Step 3 completed. Paragraph metrics saved to: ${PARAGRAPH_OUTPUT}"
        ;;
    A|a)
        echo -e "\nRunning all steps sequentially..."
        
        # Step 1
        echo -e "\n[Step 1] Running baseline LLM classification using ${OLLAMA_MODEL}..."
        
        # Check Ollama server and model availability
        check_ollama
        ensure_model "$OLLAMA_MODEL"
        
        echo "Target DPAs: ${TARGET_DPAS[*]}"
        echo "Sample ratio: ${SAMPLE_RATIO}"
        echo "Output: ${BASELINE_RESULTS}"
        
        python3 baseline_evaluation.py \
          --model "${OLLAMA_MODEL}" \
          --dpa "${DPA_CSV}" \
          --output "${OUTPUT_DIR}" \
          --target_dpas "$(IFS=','; echo "${TARGET_DPAS[*]}")" \
          --req_ids "${REQ_IDS}" \
          --max_segments "${MAX_SEGMENTS}" \
          --sample_ratio "${SAMPLE_RATIO}" \
          --seed "${SEED}" \
          $([ "$VERBOSE" = true ] && echo "--verbose")
        
        # Step 2
        echo -e "\n[Step 2] Evaluating DPA completeness..."
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
              --results "${BASELINE_RESULTS}" \
              --dpa "${DPA_CSV}" \
              --output "${TEMP_OUTPUT}" \
              --target_dpa "${TARGET_DPA}" \
              --max_segments "${MAX_SEGMENTS}" \
              $([ "$VERBOSE" = true ] && echo "--debug")
        done
        
        python3 aggregate_evaluations.py \
          --input_files "${TEMP_OUTPUTS}" \
          --output "${EVALUATION_OUTPUT}"
        
        # Step 3
        echo -e "\n[Step 3] Calculating paragraph-level metrics..."
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
              --results "${BASELINE_RESULTS}" \
              --dpa "${DPA_CSV}" \
              --evaluation "${EVALUATION_OUTPUT}" \
              --output "${TEMP_OUTPUT}" \
              --target_dpa "${TARGET_DPA}" \
              --max_segments "${MAX_SEGMENTS}" \
              $([ "$VERBOSE" = true ] && echo "--debug")
        done
        
        python3 aggregate_paragraph_metrics.py \
          --input_files "${TEMP_OUTPUTS}" \
          --output "${PARAGRAPH_OUTPUT}"
        
        echo -e "\nAll steps completed successfully!"
        echo "=========================================="
        echo "Configuration used:"
        echo "  Model: ${OLLAMA_MODEL} (Ollama)"
        echo "  Evaluation type: Baseline (Direct LLM Classification)"
        echo "  Target DPAs: ${TARGET_DPAS[*]}"
        echo "  Requirement IDs: ${REQ_IDS}"
        echo "  Max segments: ${MAX_SEGMENTS}"
        echo "  Sample ratio: ${SAMPLE_RATIO}"
        echo "  Random seed: ${SEED}"
        echo ""
        echo "Results saved to:"
        echo "  Output directory: ${OUTPUT_DIR}"
        echo "  Baseline results: ${BASELINE_RESULTS}"
        echo "  Evaluation results: ${EVALUATION_OUTPUT}"
        echo "  Paragraph metrics: ${PARAGRAPH_OUTPUT}"
        echo "=========================================="
        ;;
    Q|q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option. Please choose a valid step (1-3, A, or Q)."
        exit 1
        ;;
esac 