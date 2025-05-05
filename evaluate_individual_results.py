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
            if "satisfies(req" in line:
                results[current_req][current_segment] = "satisfies"
            elif "not_mentioned(req" in line:
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
    with open("data/requirements/ground_truth_requirements.txt", 'r') as f:
        for line in f:
            match = re.match(r'^(\d+)\.\s*(.+)$', line.strip())
            if match:
                requirement_texts[match.group(1)] = match.group(2)
    
    # Compute ground truth
    df = pd.read_csv(dpa_csv)
    dpa_df = df[df['DPA'] == 'Online 1']
    
    covered_reqs = set()
    for _, row in dpa_df.iterrows():
        for col in ["Requirement-1", "Requirement-2", "Requirement-3"]:
            if col in row and pd.notna(row[col]):
                covered_reqs.add(row[col].replace('R', ''))
    
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
        covered_count = len(covered_reqs.intersection(set(range(7, 25))))
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Summary:\n")
        f.write(f"Predicted satisfied: {satisfied_count}/{len(individual_results)}\n")
        f.write(f"Actually covered: {covered_count}/{total_required}\n")

# Run evaluation
evaluate_individual_results("semantic_results/individual_deolingo_results.txt", 
                          "data/train_set.csv",
                          "semantic_results/individual_evaluation_results.txt")
