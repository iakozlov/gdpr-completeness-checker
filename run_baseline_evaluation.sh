#!/bin/bash

# Baseline DPA Completeness Evaluation using Direct LLM Classification
# This script provides a baseline comparison to the LP-based symbolic reasoning approach

# Default values
MODEL="gemma3:27b"
DPA_FILE="data/test_set.csv"
TARGET_DPAS="Online 124,Online 126,Online 132"
REQ_IDS="all"
MAX_SEGMENTS=0
OUTPUT_DIR="results/gemma3_27b/baseline_experiments"
VERBOSE=false
STEP="4"

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
        --step)
            STEP="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Baseline DPA Completeness Evaluation using Direct LLM Classification"
            echo ""
            echo "Options:"
            echo "  --ollama_model MODEL      Ollama model to use (default: gemma3:27b)"
            echo "  --dpa_file FILE          Path to DPA CSV file (default: data/test_set.csv)"
            echo "  --target_dpas DPAS       Comma-separated target DPAs (default: 'Online 124,Online 126,Online 132')"
            echo "  --req_ids IDS            Comma-separated requirement IDs or 'all' (default: all)"
            echo "  --max_segments N         Maximum segments per DPA (0=all, default: 0)"
            echo "  --output_dir DIR         Custom output directory (default: results/gemma3_27b/baseline_experiments)"
            echo "  --verbose                Enable verbose output"
            echo "  --step STEP              Run specific step (1=baseline, 2=evaluate, 3=paragraph, 4=aggregate, all=all steps)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Steps:"
            echo "  1 - Run baseline LLM classification"
            echo "  2 - Evaluate completeness using baseline results"
            echo "  3 - Calculate paragraph-level metrics" 
            echo "  4 - Aggregate results from all DPAs"
            echo "  all - Run all steps sequentially"
            echo ""
            echo "Examples:"
            echo "  $0 --step all                                    # Run all steps"
            echo "  $0 --step 1 --ollama_model llama3.3:70b        # Run only baseline classification"
            echo "  $0 --step 4 --output_dir 'results/my_experiment' # Run only aggregation"
            echo "  $0 --req_ids '1,2,3' --verbose                 # Interactive mode"
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

# Function to check Ollama server (only needed for steps 1)
check_ollama() {
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
}

# Function to run baseline LLM classification (Step 1)
run_baseline_classification() {
    echo ""
    echo "=========================================="
    echo "STEP 1: BASELINE LLM CLASSIFICATION"
    echo "=========================================="
    
    check_ollama
    
    # Convert target DPAs to array for processing each DPA separately
    IFS=',' read -ra DPA_ARRAY <<< "$TARGET_DPAS"

    # Process each DPA
    for TARGET_DPA in "${DPA_ARRAY[@]}"; do
        # Clean DPA name
        TARGET_DPA=$(echo "$TARGET_DPA" | xargs)  # Remove leading/trailing whitespace
        DPA_CLEAN=$(echo "$TARGET_DPA" | sed 's/ /_/g')  # Replace spaces with underscores for filenames
        
        echo ""
        echo "Processing DPA: $TARGET_DPA"
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
        
        echo "Completed baseline classification for DPA: $TARGET_DPA"
    done
    
    echo "Step 1 completed. Baseline classification results saved in: $RESULTS_DIR"
}

# Function to evaluate completeness (Step 2)
run_completeness_evaluation() {
    echo ""
    echo "=========================================="
    echo "STEP 2: COMPLETENESS EVALUATION"
    echo "=========================================="
    
    # Convert target DPAs to array for processing each DPA separately
    IFS=',' read -ra DPA_ARRAY <<< "$TARGET_DPAS"

    # Process each DPA
    for TARGET_DPA in "${DPA_ARRAY[@]}"; do
        # Clean DPA name
        TARGET_DPA=$(echo "$TARGET_DPA" | xargs)
        DPA_CLEAN=$(echo "$TARGET_DPA" | sed 's/ /_/g')
        
        echo ""
        echo "Processing DPA: $TARGET_DPA"
        echo "----------------------------------------"
        
        DEOLINGO_RESULTS="$RESULTS_DIR/baseline_${DPA_CLEAN}/baseline_deolingo_results_${DPA_CLEAN}.txt"
        EVALUATION_OUTPUT="$RESULTS_DIR/evaluation_${DPA_CLEAN}.json"
        
        if [ ! -f "$DEOLINGO_RESULTS" ]; then
            echo "Error: Baseline results not found for DPA $TARGET_DPA. Run Step 1 first."
            continue
        fi
        
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
        
        echo "Completed completeness evaluation for DPA: $TARGET_DPA"
    done
    
    echo "Step 2 completed. Evaluation results saved in: $RESULTS_DIR"
}

# Function to calculate paragraph metrics (Step 3)
run_paragraph_metrics() {
    echo ""
    echo "=========================================="
    echo "STEP 3: PARAGRAPH-LEVEL METRICS"
    echo "=========================================="
    
    # Convert target DPAs to array for processing each DPA separately
    IFS=',' read -ra DPA_ARRAY <<< "$TARGET_DPAS"

    # Process each DPA
    for TARGET_DPA in "${DPA_ARRAY[@]}"; do
        # Clean DPA name
        TARGET_DPA=$(echo "$TARGET_DPA" | xargs)
        DPA_CLEAN=$(echo "$TARGET_DPA" | sed 's/ /_/g')
        
        echo ""
        echo "Processing DPA: $TARGET_DPA"
        echo "----------------------------------------"
        
        DEOLINGO_RESULTS="$RESULTS_DIR/baseline_${DPA_CLEAN}/baseline_deolingo_results_${DPA_CLEAN}.txt"
        PARAGRAPH_OUTPUT="$RESULTS_DIR/paragraph_metrics_${DPA_CLEAN}.json"
        
        if [ ! -f "$DEOLINGO_RESULTS" ]; then
            echo "Error: Baseline results not found for DPA $TARGET_DPA. Run Step 1 first."
            continue
        fi
        
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
            continue
        fi
        
        echo "Completed paragraph metrics for DPA: $TARGET_DPA"
    done
    
    echo "Step 3 completed. Paragraph metrics saved in: $RESULTS_DIR"
}

# Function to aggregate results (Step 4)
run_aggregation() {
    echo ""
    echo "=========================================="
    echo "STEP 4: AGGREGATING RESULTS"
    echo "=========================================="

    # Collect individual evaluation files for aggregation
    EVAL_FILES=""
    PARAGRAPH_FILES=""

    # Convert target DPAs to array for proper handling of spaces in DPA names
    IFS=',' read -ra DPA_ARRAY <<< "$TARGET_DPAS"

    for TARGET_DPA in "${DPA_ARRAY[@]}"; do
        # Clean DPA name
        TARGET_DPA=$(echo "$TARGET_DPA" | xargs)  # Remove leading/trailing whitespace
        DPA_CLEAN=$(echo "$TARGET_DPA" | sed 's/ /_/g')  # Replace spaces with underscores for filenames
        
        EVAL_FILE="$RESULTS_DIR/evaluation_${DPA_CLEAN}.json"
        PARAGRAPH_FILE="$RESULTS_DIR/paragraph_metrics_${DPA_CLEAN}.json"
        
        echo "Looking for evaluation file: $EVAL_FILE"
        
        if [ -f "$EVAL_FILE" ]; then
            echo "Found: $EVAL_FILE"
            if [ -z "$EVAL_FILES" ]; then
                EVAL_FILES="$EVAL_FILE"
            else
                EVAL_FILES="$EVAL_FILES,$EVAL_FILE"
            fi
        else
            echo "Not found: $EVAL_FILE"
        fi
        
        if [ -f "$PARAGRAPH_FILE" ]; then
            echo "Found: $PARAGRAPH_FILE"
            if [ -z "$PARAGRAPH_FILES" ]; then
                PARAGRAPH_FILES="$PARAGRAPH_FILE"
            else
                PARAGRAPH_FILES="$PARAGRAPH_FILES,$PARAGRAPH_FILE"
            fi
        else
            echo "Not found: $PARAGRAPH_FILE"
        fi
    done

    # Generate aggregated evaluation results
    if [ -n "$EVAL_FILES" ]; then
        echo ""
        echo "Aggregating evaluation results..."
        echo "Input files: $EVAL_FILES"
        AGGREGATED_EVAL="$RESULTS_DIR/aggregated_evaluation_results.json"
        
        python3 aggregate_evaluations.py \
            --input_files "$EVAL_FILES" \
            --output "$AGGREGATED_EVAL"
        
        if [ $? -eq 0 ]; then
            echo "Aggregated evaluation results saved to: $AGGREGATED_EVAL"
        else
            echo "Warning: Failed to generate aggregated evaluation results"
        fi
    else
        echo "No evaluation files found to aggregate. Make sure Step 2 has been completed."
    fi

    # Generate aggregated paragraph metrics
    if [ -n "$PARAGRAPH_FILES" ]; then
        echo ""
        echo "Aggregating paragraph metrics..."
        echo "Input files: $PARAGRAPH_FILES"
        AGGREGATED_PARAGRAPH="$RESULTS_DIR/aggregated_paragraph_metrics.json"
        
        python3 aggregate_paragraph_metrics.py \
            --input_files "$PARAGRAPH_FILES" \
            --output "$AGGREGATED_PARAGRAPH"
        
        if [ $? -eq 0 ]; then
            echo "Aggregated paragraph metrics saved to: $AGGREGATED_PARAGRAPH"
        else
            echo "Warning: Failed to generate aggregated paragraph metrics"
        fi
    else
        echo "No paragraph metrics files found to aggregate. Make sure Step 3 has been completed."
    fi
    
    echo "Step 4 completed. Aggregated results saved in: $RESULTS_DIR"
}

# Main execution logic
if [ -n "$STEP" ]; then
    # Run specific step
    case $STEP in
        1)
            run_baseline_classification
            ;;
        2)
            run_completeness_evaluation
            ;;
        3)
            run_paragraph_metrics
            ;;
        4)
            run_aggregation
            ;;
        all)
            run_baseline_classification
            run_completeness_evaluation
            run_paragraph_metrics
            run_aggregation
            ;;
        *)
            echo "Invalid step: $STEP. Use 1, 2, 3, 4, or 'all'"
            exit 1
            ;;
    esac
else
    # Interactive menu
    while true; do
        echo ""
        echo "=========================================="
        echo "SELECT STEP TO EXECUTE"
        echo "=========================================="
        echo "1) Run baseline LLM classification"
        echo "2) Evaluate completeness using baseline results"
        echo "3) Calculate paragraph-level metrics"
        echo "4) Aggregate results from all DPAs"
        echo "5) Run all steps sequentially"
        echo "q) Quit"
        echo ""
        echo "Current settings:"
        echo "  Model: $MODEL"
        echo "  Target DPAs: $TARGET_DPAS"
        echo "  Output dir: $RESULTS_DIR"
        echo ""
        read -p "Choose an option [1-5, q]: " choice
        
        case $choice in
            1)
                run_baseline_classification
                ;;
            2)
                run_completeness_evaluation
                ;;
            3)
                run_paragraph_metrics
                ;;
            4)
                run_aggregation
                ;;
            5)
                run_baseline_classification
                run_completeness_evaluation
                run_paragraph_metrics
                run_aggregation
                echo ""
                echo "All steps completed successfully!"
                ;;
            [qQ])
                echo "Exiting..."
                exit 0
                ;;
            *)
                echo "Invalid option. Please choose 1-5 or q."
                ;;
        esac
    done
fi

echo ""
echo "=========================================="
echo "BASELINE EVALUATION COMPLETED"
echo "=========================================="
echo "All results saved in: $RESULTS_DIR"
echo ""
echo "Generated files:"
echo "- baseline_[DPA]/: Direct LLM classification results"
echo "- evaluation_[DPA].json: Individual DPA completeness evaluation metrics"
echo "- paragraph_metrics_[DPA].json: Individual DPA paragraph-level analysis"
echo "- aggregated_evaluation_results.json: Combined evaluation metrics across all DPAs"
echo "- aggregated_paragraph_metrics.json: Combined paragraph metrics across all DPAs"
echo ""
echo "To view results:"
echo "  cat $RESULTS_DIR/evaluation_*.json"
echo "  python3 -m json.tool $RESULTS_DIR/evaluation_*.json"
echo "  python3 -m json.tool $RESULTS_DIR/aggregated_evaluation_results.json" 