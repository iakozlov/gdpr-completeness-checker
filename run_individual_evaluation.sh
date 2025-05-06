#!/bin/bash
# run_individual_evaluation.sh - Fixed version
# Script to run evaluation on individual LP files

set -e  # Exit on any error

# Configuration
DPA_CSV="data/train_set.csv"
REQUIREMENTS_FILE="data/processed/requirements_symbolic.json"
MODEL_PATH="models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf"
TARGET_DPA="Online 1"
MAX_SEGMENTS=10
OUTPUT_DIR="semantic_results/individual_lp_files"
RESULTS_FILE="semantic_results/individual_deolingo_results.txt"
EVALUATION_OUTPUT="semantic_results/individual_evaluation_results.txt"

echo "========== Individual LP File Evaluation =========="
echo "Target DPA: ${TARGET_DPA}"
echo "=================================================="

# Function to run step 1: Translate DPA segments
run_step_1() {
    echo -e "\n[Step 1] Translating DPA segments to deontic logic..."
    python translate_dpa_semantic.py \
      --dpa ${DPA_CSV} \
      --model ${MODEL_PATH} \
      --output "semantic_results/dpa_deontic.json" \
      --target "${TARGET_DPA}" \
      --max_segments ${MAX_SEGMENTS}
    echo "Step 1 completed."
}

# Function to run step 2: Generate individual LP files
run_step_2() {
    echo -e "\n[Step 2] Generating individual LP files for requirement 6..."
    python generate_individual_lp_files.py \
      --requirements ${REQUIREMENTS_FILE} \
      --dpa_segments "semantic_results/dpa_deontic.json" \
      --model ${MODEL_PATH} \
      --output ${OUTPUT_DIR}
    echo "Step 2 completed."
}

# Function to run step 3: Process all LP files
run_step_3() {
    echo -e "\n[Step 3] Running Deolingo on all LP files..."
    
    # Create output file for results
    echo "" > ${RESULTS_FILE}
    
    # Process all LP files for requirement 6
    echo "Processing LP files..."
    
    # The files are now in: OUTPUT_DIR/req_6/dpa_segment_*.lp
    REQ_DIR="${OUTPUT_DIR}/req_6"
    
    # Check if directory exists
    if [ ! -d "${REQ_DIR}" ]; then
        echo "Error: Directory ${REQ_DIR} not found. Run step 2 first."
        return 1
    fi
    
    # Find all LP files in the req_6 directory
    find "${REQ_DIR}" -name "dpa_segment_*.lp" | sort | while read lp_file; do
        # Extract segment ID from filename
        segment_id=$(basename "${lp_file}" .lp | sed 's/dpa_segment_//')
        
        echo "Processing Requirement 6, DPA Segment ${segment_id}..." | tee -a ${RESULTS_FILE}
        
        # Run deolingo with error handling
        if deolingo "${lp_file}" >> ${RESULTS_FILE} 2>&1; then
            echo "Success: Requirement 6, Segment ${segment_id}" >> ${RESULTS_FILE}
        else
            echo "Error: Syntax error in Requirement 6, Segment ${segment_id}" >> ${RESULTS_FILE}
        fi
        
        echo "--------------------------------------------------" | tee -a ${RESULTS_FILE}
    done
    
    echo "Step 3 completed. Results saved in: ${RESULTS_FILE}"
}

# Function to run step 4: Evaluate results
run_step_4() {
    echo -e "\n[Step 4] Evaluating individual results..."
    
    # Create evaluation script for individual results
    cat > evaluate_individual_results.py << 'EOF'
import re
import pandas as pd
import json
from collections import defaultdict

def parse_individual_results(results_file):
    """Parse individual LP file results."""
    results = defaultdict(lambda: defaultdict(dict))
    
    with open(results_file, 'r') as f:
        content = f.readlines()
    
    current_req = None
    current_segment = None
    
    for line in content:
        # Extract requirement and segment IDs
        req_match = re.search(r'Requirement (\d+), DPA Segment (\d+)', line)
        if req_match:
            current_req = req_match.group(1)
            current_segment = req_match.group(2)
            continue
        
        # Extract results
        if current_req and current_segment:
            if "satisfies(req)" in line:
                results[current_req][current_segment] = "satisfies"
            elif "not_mentioned(req)" in line:
                results[current_req][current_segment] = "not_mentioned"
            elif "Syntax error" in line or "Error" in line:
                results[current_req][current_segment] = "error"
    
    # Aggregate results per requirement
    aggregated_results = {}
    for req_id, segments in results.items():
        if segments:  # If we have any results for this requirement
            # A requirement is satisfied if any segment satisfies it
            is_satisfied = any(status == "satisfies" for status in segments.values())
            aggregated_results[req_id] = "satisfies" if is_satisfied else "not_mentioned"
        else:
            aggregated_results[req_id] = "not_mentioned"
    
    return aggregated_results

def evaluate_individual_results(results_file, dpa_csv, output_file):
    """Evaluate individual LP file results."""
    # Parse individual results
    individual_results = parse_individual_results(results_file)
    
    # Load requirements texts
    requirement_texts = {}
    try:
        with open("data/requirements/ground_truth_requirements.txt", 'r') as f:
            for line in f:
                match = re.match(r'^(\d+)\.\s*(.+)$', line.strip())
                if match:
                    requirement_texts[match.group(1)] = match.group(2)
    except:
        print("Could not load requirement texts, using placeholders")
        requirement_texts['6'] = "The processor shall take all measures required pursuant to Article 32..."
    
    # Compute ground truth
    df = pd.read_csv(dpa_csv)
    dpa_df = df[df['DPA'] == 'Online 1']
    
    covered_reqs = set()
    for _, row in dpa_df.iterrows():
        for col in ["Requirement-1", "Requirement-2", "Requirement-3"]:
            if col in row and pd.notna(row[col]):
                req_num = row[col].replace('R', '')
                covered_reqs.add(req_num)
    
    # Generate evaluation report
    with open(output_file, 'w') as f:
        f.write("========== Individual LP File Evaluation Results ==========\n\n")
        
        f.write("Requirement-Level Results:\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Req ID':<8} {'Status':<15} {'Ground Truth':<15} {'Requirement Text'}\n")
        f.write("-" * 120 + "\n")
        
        for req_id in sorted(individual_results.keys(), key=int):
            status = individual_results[req_id].upper()
            req_name = f"R{req_id}"
            gt_status = "COVERED" if req_id in covered_reqs else "NOT COVERED"
            req_text = requirement_texts.get(req_id, "Text not found")[:80]
            
            f.write(f"{req_name:<8} {status:<15} {gt_status:<15} {req_text}\n")
        
        # Summary
        satisfied_count = sum(1 for status in individual_results.values() if status == "satisfies")
        total_required = len(range(7, 25))
        covered_count = len(covered_reqs.intersection(set(str(i) for i in range(7, 25))))
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Summary:\n")
        f.write(f"Predicted satisfied: {satisfied_count}/{len(individual_results)}\n")
        f.write(f"Actually covered: {covered_count}/{total_required}\n")
        
        # Special check for requirement 6
        if '6' in individual_results:
            segment_26_status = "Unknown"
            
            # Look through the original results to find segment 26
            with open(results_file, 'r') as rf:
                content = rf.read()
                if "Requirement 6, DPA Segment 26" in content:
                    if "satisfies(req)" in content:
                        segment_26_status = "SATISFIES"
                    elif "not_mentioned(req)" in content:
                        segment_26_status = "NOT_MENTIONED"
                    else:
                        segment_26_status = "ERROR"
            
            f.write(f"\nRequirement 6, Segment 26 Status: {segment_26_status}\n")

# Run evaluation
evaluate_individual_results("semantic_results/individual_deolingo_results.txt", 
                          "data/train_set.csv",
                          "semantic_results/individual_evaluation_results.txt")
EOF
    
    # Run the evaluation script
    python evaluate_individual_results.py
    
    # Display results
    echo "=================================================="
    echo "Evaluation Results:"
    tail -n 15 ${EVALUATION_OUTPUT}
    echo "=================================================="
    echo "Step 4 completed."
}

# Main execution
echo "Select step to run:"
echo "1. Translate DPA segments"
echo "2. Generate individual LP files"
echo "3. Run Deolingo on all LP files"
echo "4. Evaluate results"
echo "A. Run all steps"
echo "Q. Quit"

read -p "Enter choice: " choice

case ${choice} in
    1) run_step_1 ;;
    2) run_step_2 ;;
    3) run_step_3 ;;
    4) run_step_4 ;;
    [aA]) 
        run_step_1
        run_step_2 
        run_step_3
        run_step_4
        ;;
    [qQ]) echo "Exiting..."; exit 0 ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

echo "Done!"