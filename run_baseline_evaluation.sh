#!/bin/bash

# Baseline DPA Completeness Evaluation using Direct LLM Classification
# This script provides a baseline comparison to the LP-based symbolic reasoning approach

# Default values
MODEL="llama3.3:70b"
DPA_FILE="data/train_set.csv"
TARGET_DPAS="Online 124,Online 132"
REQ_IDS="all"
MAX_SEGMENTS=0
OUTPUT_DIR=""
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ollama_model)
            MODEL="$2"
            shift 2
            ;;
        --dpa_file)
            DPA_FILE="$2"
            shift 2
            ;;
        --target_dpas)
            TARGET_DPAS="$2"
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
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
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
            echo "  --ollama_model MODEL      Ollama model to use (default: llama3.3:70b)"
            echo "  --dpa_file FILE          Path to DPA CSV file (default: data/train_set.csv)"
            echo "  --target_dpas DPAS       Comma-separated target DPAs (default: 'Online 124,Online 132')"
            echo "  --req_ids IDS            Comma-separated requirement IDs or 'all' (default: all)"
            echo "  --max_segments N         Maximum segments per DPA (0=all, default: 0)"
            echo "  --output_dir DIR         Custom output directory (default: results/baseline_experiments/TIMESTAMP_MODEL)"
            echo "  --verbose                Enable verbose output"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --ollama_model llama3.3:70b --target_dpas 'Online 124'"
            echo "  $0 --req_ids '1,2,3' --verbose"
            echo "  $0 --output_dir 'my_custom_results' --target_dpas 'Online 124'"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

# Set results directory - use custom dir if provided, otherwise create timestamped directory
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RESULTS_DIR="results/baseline_experiments/${TIMESTAMP}_${MODEL//:/}"
else
    RESULTS_DIR="$OUTPUT_DIR"
fi
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "BASELINE DPA COMPLETENESS EVALUATION"
echo "=========================================="
echo "Model: $MODEL"
echo "DPA File: $DPA_FILE"
echo "Target DPAs: $TARGET_DPAS"
echo "Requirements: $REQ_IDS"
echo "Max segments: $MAX_SEGMENTS"
echo "Results directory: $RESULTS_DIR"
echo "=========================================="

# Check if Ollama is running
echo "Checking Ollama server status..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Error: Ollama server is not running. Please start it first with: ollama serve"
    exit 1
fi

echo "Ollama server is running. Checking model availability..."
if ! ollama list | grep -q "$MODEL"; then
    echo "Model $MODEL not found. Pulling it now..."
    ollama pull "$MODEL"
fi

# Convert target DPAs to array for processing each DPA separately
IFS=',' read -ra DPA_ARRAY <<< "$TARGET_DPAS"

# Process each DPA
for TARGET_DPA in "${DPA_ARRAY[@]}"; do
    # Clean DPA name
    TARGET_DPA=$(echo "$TARGET_DPA" | xargs)  # Remove leading/trailing whitespace
    DPA_CLEAN=$(echo "$TARGET_DPA" | sed 's/ /_/g')  # Replace spaces with underscores for filenames
    
    echo ""
    echo "=========================================="
    echo "PROCESSING DPA: $TARGET_DPA"
    echo "=========================================="
    
    # Step 1: Run baseline LLM classification
    echo ""
    echo "Step 1: Running baseline LLM classification..."
    echo "----------------------------------------"
    
    BASELINE_OUTPUT="$RESULTS_DIR/baseline_${DPA_CLEAN}"
    
    python3 baseline_evaluation.py \
        --model "$MODEL" \
        --dpa "$DPA_FILE" \
        --output "$BASELINE_OUTPUT" \
        --target_dpas "$TARGET_DPA" \
        --req_ids "$REQ_IDS" \
        --max_segments "$MAX_SEGMENTS" \
        $([ "$VERBOSE" = true ] && echo "--verbose")
    
    if [ $? -ne 0 ]; then
        echo "Error: Baseline classification failed for DPA: $TARGET_DPA"
        continue
    fi
    
    # Step 2: Evaluate completeness using the same evaluation script
    echo ""
    echo "Step 2: Evaluating completeness..."
    echo "----------------------------------------"
    
    DEOLINGO_RESULTS="$BASELINE_OUTPUT/baseline_deolingo_results_${DPA_CLEAN}.txt"
    EVALUATION_OUTPUT="$RESULTS_DIR/evaluation_${DPA_CLEAN}.json"
    
    python3 evaluate_completeness.py \
        --results "$DEOLINGO_RESULTS" \
        --dpa "$DPA_FILE" \
        --output "$EVALUATION_OUTPUT" \
        --target_dpa "$TARGET_DPA" \
        --req_ids "$REQ_IDS" \
        --max_segments "$MAX_SEGMENTS" \
        $([ "$VERBOSE" = true ] && echo "--debug")
    
    if [ $? -ne 0 ]; then
        echo "Error: Completeness evaluation failed for DPA: $TARGET_DPA"
        continue
    fi
    
    # Step 3: Calculate paragraph-level metrics
    echo ""
    echo "Step 3: Calculating paragraph-level metrics..."
    echo "----------------------------------------"
    
    PARAGRAPH_OUTPUT="$RESULTS_DIR/paragraph_metrics_${DPA_CLEAN}.json"
    
    python3 paragraph_metrics.py \
        --results "$DEOLINGO_RESULTS" \
        --dpa "$DPA_FILE" \
        --output "$PARAGRAPH_OUTPUT" \
        --target_dpa "$TARGET_DPA" \
        --req_ids "$REQ_IDS" \
        --max_segments "$MAX_SEGMENTS" \
        $([ "$VERBOSE" = true ] && echo "--debug")
    
    if [ $? -ne 0 ]; then
        echo "Warning: Paragraph metrics calculation failed for DPA: $TARGET_DPA (continuing...)"
    fi
    
    echo ""
    echo "Completed processing for DPA: $TARGET_DPA"
    echo "Results saved in: $RESULTS_DIR/"
done

echo ""
echo "=========================================="
echo "BASELINE EVALUATION COMPLETED"
echo "=========================================="
echo "All results saved in: $RESULTS_DIR"
echo ""
echo "Generated files:"
echo "- baseline_[DPA]/: Direct LLM classification results"
echo "- evaluation_[DPA].json: Completeness evaluation metrics"
echo "- paragraph_metrics_[DPA].json: Paragraph-level analysis"
echo ""
echo "To view results:"
echo "  cat $RESULTS_DIR/evaluation_*.json"
echo "  python3 -m json.tool $RESULTS_DIR/evaluation_*.json" 