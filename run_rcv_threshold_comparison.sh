#!/bin/bash
# run_rcv_threshold_comparison.sh
# Script to run RCV approach with different similarity thresholds for comparison

# Note: Removed 'set -e' to allow script to continue if one threshold fails

# Configuration
THRESHOLDS=(0.3 0.5 0.7 0.8 0.9 0.95)
BASE_OUTPUT_DIR="results/rcv_approach/threshold_comparison"
OLLAMA_MODEL="qwen2.5:32b"
EMBEDDING_MODEL="all-MiniLM-L6-v2"
TARGET_DPAS=("Online 124" "Online 132" "Online 54")
REQUIREMENTS_REPRESENTATION="deontic_ai"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            OLLAMA_MODEL="$2"
            shift 2
            ;;
        --embedding_model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --target_dpas)
            IFS=',' read -ra TARGET_DPAS <<< "$2"
            shift 2
            ;;
        --requirements)
            REQUIREMENTS_REPRESENTATION="$2"
            shift 2
            ;;
        --output_dir)
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --thresholds)
            IFS=',' read -ra THRESHOLDS <<< "$2"
            shift 2
            ;;
        --help|-h)
            echo "RCV Threshold Comparison Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL                 Ollama model for verification (default: qwen2.5:32b)"
            echo "  --embedding_model MODEL       Embedding model for classification (default: all-MiniLM-L6-v2)"
            echo "  --target_dpas DPA1,DPA2       Comma-separated list of DPAs (default: Online 124,Online 132,Online 54)"
            echo "  --requirements REPR           Requirements representation (default: deontic_ai)"
            echo "  --output_dir DIR              Base output directory (default: results/rcv_approach/threshold_comparison)"
            echo "  --thresholds T1,T2,T3         Comma-separated similarity thresholds (default: 0.3,0.5,0.7,0.8,0.9,0.95)"
            echo "  --help, -h                    Show this help message"
            echo ""
            echo "This script will run the RCV approach with the specified similarity thresholds."
            echo "Results will be saved to separate subdirectories for each threshold."
            exit 0
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

# Create base output directory
mkdir -p "${BASE_OUTPUT_DIR}"

echo "========== RCV Threshold Comparison =========="
echo "Testing similarity thresholds: ${THRESHOLDS[*]}"
echo "Verification Model (LLM): ${OLLAMA_MODEL}"
echo "Classification Model (Embedding): ${EMBEDDING_MODEL}"
echo "Target DPAs: ${TARGET_DPAS[*]}"
echo "Base Output Directory: ${BASE_OUTPUT_DIR}"
echo "=============================================="

# Summary file to collect all results
SUMMARY_FILE="${BASE_OUTPUT_DIR}/threshold_comparison_summary.json"
echo "{" > ${SUMMARY_FILE}
echo "  \"comparison_metadata\": {" >> ${SUMMARY_FILE}
echo "    \"timestamp\": \"$(date -Iseconds)\"," >> ${SUMMARY_FILE}
echo "    \"ollama_model\": \"${OLLAMA_MODEL}\"," >> ${SUMMARY_FILE}
echo "    \"embedding_model\": \"${EMBEDDING_MODEL}\"," >> ${SUMMARY_FILE}
echo "    \"thresholds_tested\": [$(printf '%.2f,' "${THRESHOLDS[@]}" | sed 's/,$//')]," >> ${SUMMARY_FILE}
echo "    \"target_dpas\": [$(printf '\"%s\",' "${TARGET_DPAS[@]}" | sed 's/,$//')]" >> ${SUMMARY_FILE}
echo "  }," >> ${SUMMARY_FILE}
echo "  \"results\": {" >> ${SUMMARY_FILE}

# Counter for comma separation in JSON
threshold_count=0

# Run the pipeline for each threshold
for threshold in "${THRESHOLDS[@]}"; do
    log "Starting evaluation with similarity threshold: ${threshold}"
    
    # Create threshold-specific output directory
    THRESHOLD_OUTPUT_DIR="${BASE_OUTPUT_DIR}/threshold_${threshold}"
    mkdir -p "${THRESHOLD_OUTPUT_DIR}"
    
    log "Output directory: ${THRESHOLD_OUTPUT_DIR}"
    
    # Run the RCV pipeline with current threshold
    log "Running RCV pipeline with threshold ${threshold}..."
    
    if ./run_dpa_completeness_rcv.sh \
        --step A \
        --model "${OLLAMA_MODEL}" \
        --embedding_model "${EMBEDDING_MODEL}" \
        --similarity_threshold "${threshold}" \
        --requirements "${REQUIREMENTS_REPRESENTATION}" \
        --target_dpas "$(IFS=','; echo "${TARGET_DPAS[*]}")" \
        --output_dir "${THRESHOLD_OUTPUT_DIR}"; then
        log "Completed evaluation with threshold ${threshold}"
    else
        log "ERROR: Failed to complete evaluation with threshold ${threshold}"
        # Continue with next threshold instead of exiting
    fi
    
    # Extract key metrics from the evaluation results
    EVAL_FILE="${THRESHOLD_OUTPUT_DIR}/evaluation_results_$(echo ${OLLAMA_MODEL} | tr ':' '_' | tr '/' '_')_rcv.json"
    
    if [[ -f "${EVAL_FILE}" ]]; then
        # Add comma separator for JSON (except for first entry)
        if [[ ${threshold_count} -gt 0 ]]; then
            echo "," >> ${SUMMARY_FILE}
        fi
        
        # Extract key metrics using jq if available, otherwise use basic parsing
        if command -v jq &> /dev/null; then
            echo "    \"${threshold}\": {" >> ${SUMMARY_FILE}
            echo "      \"evaluation_file\": \"${EVAL_FILE}\"," >> ${SUMMARY_FILE}
            
            # Try to extract metrics with error handling
            if REQ_METRICS=$(jq '.aggregated_metrics.requirements.overall' "${EVAL_FILE}" 2>/dev/null); then
                echo "      \"requirement_metrics\": ${REQ_METRICS}," >> ${SUMMARY_FILE}
            else
                echo "      \"requirement_metrics\": null," >> ${SUMMARY_FILE}
            fi
            
            if SEG_METRICS=$(jq '.aggregated_metrics.segments.overall' "${EVAL_FILE}" 2>/dev/null); then
                echo "      \"segment_metrics\": ${SEG_METRICS}," >> ${SUMMARY_FILE}
            else
                echo "      \"segment_metrics\": null," >> ${SUMMARY_FILE}
            fi
            
            echo "      \"completeness\": {" >> ${SUMMARY_FILE}
            
            if GT_COMPLETE=$(jq '.ground_truth.is_complete' "${EVAL_FILE}" 2>/dev/null); then
                echo "        \"ground_truth_complete\": ${GT_COMPLETE}," >> ${SUMMARY_FILE}
            else
                echo "        \"ground_truth_complete\": null," >> ${SUMMARY_FILE}
            fi
            
            if PRED_COMPLETE=$(jq '.prediction.is_complete' "${EVAL_FILE}" 2>/dev/null); then
                echo "        \"prediction_complete\": ${PRED_COMPLETE}," >> ${SUMMARY_FILE}
            else
                echo "        \"prediction_complete\": null," >> ${SUMMARY_FILE}
            fi
            
            if AGREEMENT=$(jq '.agreement' "${EVAL_FILE}" 2>/dev/null); then
                echo "        \"agreement\": ${AGREEMENT}" >> ${SUMMARY_FILE}
            else
                echo "        \"agreement\": null" >> ${SUMMARY_FILE}
            fi
            
            echo "      }" >> ${SUMMARY_FILE}
            echo "    }" >> ${SUMMARY_FILE}
        else
            # Fallback without jq
            echo "    \"${threshold}\": {" >> ${SUMMARY_FILE}
            echo "      \"evaluation_file\": \"${EVAL_FILE}\"," >> ${SUMMARY_FILE}
            echo "      \"note\": \"Install jq for detailed metrics extraction\"" >> ${SUMMARY_FILE}
            echo "    }" >> ${SUMMARY_FILE}
        fi
        
        log "Extracted metrics for threshold ${threshold}"
    else
        log "Warning: Evaluation file not found: ${EVAL_FILE}"
        if [[ ${threshold_count} -gt 0 ]]; then
            echo "," >> ${SUMMARY_FILE}
        fi
        echo "    \"${threshold}\": {" >> ${SUMMARY_FILE}
        echo "      \"error\": \"Evaluation file not found\"" >> ${SUMMARY_FILE}
        echo "    }" >> ${SUMMARY_FILE}
    fi
    
    ((threshold_count++))
    
    log "Completed processing for threshold ${threshold}"
    echo "=============================================="
done

# Close JSON structure
echo "" >> ${SUMMARY_FILE}
echo "  }" >> ${SUMMARY_FILE}
echo "}" >> ${SUMMARY_FILE}

echo ""
echo "========== Threshold Comparison Complete =========="
echo "Results saved to: ${BASE_OUTPUT_DIR}"
echo "Summary file: ${SUMMARY_FILE}"
echo ""
echo "Individual results:"
for threshold in "${THRESHOLDS[@]}"; do
    THRESHOLD_DIR="${BASE_OUTPUT_DIR}/threshold_${threshold}"
    if [[ -d "${THRESHOLD_DIR}" ]]; then
        echo "  Threshold ${threshold}: ${THRESHOLD_DIR}"
    fi
done
echo ""

# Display summary if jq is available
if command -v jq &> /dev/null && [[ -f "${SUMMARY_FILE}" ]]; then
    echo "Quick Summary:"
    echo "=============="
    for threshold in "${THRESHOLDS[@]}"; do
        if jq -e ".results.\"${threshold}\".requirement_metrics" "${SUMMARY_FILE}" > /dev/null 2>&1; then
            req_f1=$(jq -r ".results.\"${threshold}\".requirement_metrics.f1_score" "${SUMMARY_FILE}")
            seg_f1=$(jq -r ".results.\"${threshold}\".segment_metrics.f1_score" "${SUMMARY_FILE}")
            agreement=$(jq -r ".results.\"${threshold}\".completeness.agreement" "${SUMMARY_FILE}")
            echo "  Threshold ${threshold}: Req F1=${req_f1}, Seg F1=${seg_f1}, Agreement=${agreement}"
        else
            echo "  Threshold ${threshold}: Data not available"
        fi
    done
    echo ""
fi

log "All evaluations completed successfully!" 