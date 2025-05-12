#!/bin/bash
# run_baseline.sh
# Baseline script to run the DPA completeness evaluation pipeline
# Evaluates completeness of DPAs against requirements using direct LLM comparison

set -e  # Exit on any error

# Configuration
DPA_CSV="../data/train_set.csv"
REQUIREMENTS_FILE="../data/requirements/ground_truth_requirements.txt"
MODEL_PATH="../models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"
OUTPUT_DIR="results"
BASELINE_RESULTS="${OUTPUT_DIR}/baseline_evaluation_results.json"
METRICS_OUTPUT="${OUTPUT_DIR}/baseline_metrics.json"
TARGET_DPA="Online 1"
REQ_IDS="all"  # Process all requirements by default
MAX_SEGMENTS=0  # Process all segments by default
STEP="all"      # Run all steps by default
DETAILED=false  # Don't show detailed metrics by default

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
    --target_dpa)
      TARGET_DPA="$2"
      shift 2
      ;;
    --step)
      STEP="$2"
      shift 2
      ;;
    --baseline_results)
      BASELINE_RESULTS="$2"
      shift 2
      ;;
    --metrics_output)
      METRICS_OUTPUT="$2"
      shift 2
      ;;
    --detailed)
      DETAILED=true
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

# Show usage information
show_usage() {
  echo "Usage: ./run_baseline.sh [options]"
  echo
  echo "Options:"
  echo "  --step <name>          Specify which step to run ('evaluation', 'baseline', or 'all')"
  echo "  --req_ids <ids>        Comma-separated list of requirement IDs or 'all'"
  echo "  --max_segments <num>   Maximum number of segments to process (0 means all)"
  echo "  --target_dpa <name>    Target DPA name"
  echo "  --baseline_results <path> Path to baseline results JSON (for evaluation step)"
  echo "  --metrics_output <path>   Path for metrics output (for evaluation step)"
  echo "  --detailed             Show detailed metrics for each requirement"
  echo
  echo "Examples:"
  echo "  ./run_baseline.sh                          # Run all steps with defaults"
  echo "  ./run_baseline.sh --step baseline          # Run only baseline evaluation"
  echo "  ./run_baseline.sh --step evaluation        # Run only metrics evaluation" 
  echo "  ./run_baseline.sh --step evaluation --detailed   # Run evaluation with detailed metrics"
  echo "  ./run_baseline.sh --req_ids 1,2,3 --max_segments 10  # Run with specific requirements and segments"
}

# Display header
echo "========== Baseline DPA Completeness Checker =========="
echo "Using direct LLM comparison for requirement satisfaction"

# Run baseline step if requested
run_baseline() {
  echo "Evaluating DPA '${TARGET_DPA}'"
  echo "Focus on requirement(s): ${REQ_IDS}"
  echo "Using ${MAX_SEGMENTS} segments"
  echo "======================================================"

  echo -e "\n[Step 1] Running baseline evaluation..."
  python src/baseline_evaluator.py \
    --requirements ${REQUIREMENTS_FILE} \
    --dpa ${DPA_CSV} \
    --model ${MODEL_PATH} \
    --output ${BASELINE_RESULTS} \
    --target_dpa "${TARGET_DPA}" \
    --max_segments ${MAX_SEGMENTS} \
    --req_ids "${REQ_IDS}"

  echo "Baseline evaluation completed. Results saved to: ${BASELINE_RESULTS}"
}

# Run evaluation step if requested
run_evaluation() {
  echo -e "\n[Step 2] Calculating evaluation metrics..."
  
  # Check if baseline results file exists
  if [ ! -f "${BASELINE_RESULTS}" ]; then
    echo "Error: Baseline results file not found at: ${BASELINE_RESULTS}"
    echo "Run the baseline step first or specify the correct file with --baseline_results"
    exit 1
  fi
  
  EVAL_CMD="python src/evaluate_baseline.py \
    --baseline_results ${BASELINE_RESULTS} \
    --dpa ${DPA_CSV} \
    --output ${METRICS_OUTPUT}"
  
  # Add detailed flag if requested
  if [ "${DETAILED}" = true ]; then
    EVAL_CMD="${EVAL_CMD} --detailed"
  fi
  
  # Run the evaluation command
  eval "${EVAL_CMD}"

  echo "Evaluation metrics calculated. Results saved to: ${METRICS_OUTPUT}"
}

# Run the specified step(s)
case ${STEP} in
  baseline)
    run_baseline
    ;;
  evaluation)
    echo "Running evaluation metrics calculation only"
    echo "Using baseline results from: ${BASELINE_RESULTS}"
    if [ "${DETAILED}" = true ]; then
      echo "Showing detailed metrics for each requirement"
    fi
    echo "======================================================"
    run_evaluation
    ;;
  all)
    run_baseline
    run_evaluation
    ;;
  help)
    show_usage
    exit 0
    ;;
  *)
    echo "Error: Unknown step '${STEP}'"
    echo "Valid steps are: 'baseline', 'evaluation', 'all', or 'help'"
    exit 1
    ;;
esac
